import os
from random import randint
import uuid

from quinine import QuinineArgumentParser # Parses config files using a defined schema
from tqdm import tqdm  # Progress bar utility
import torch
import yaml # Used to save the config file to disk

from eval import get_run_metrics  # Computes evaluation metrics after training
from tasks import get_task_sampler  # Returns the appropriate task sampler
from samplers import get_data_sampler  # Returns the data sampler (e.g., Gaussian)
from curriculum import Curriculum  # Controls gradual increase in task difficulty
from schema import schema  # Defines the structure of config files
from models import build_model  # Constructs model based on config

import wandb  # Weights & Biases for logging experiment metadata and metrics

# Enable faster training by allowing cuDNN to benchmark optimal conv algorithms
torch.backends.cudnn.benchmark = True


# Performs a single gradient update step (forward pass, loss, backprop, optimizer step)
def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()  # Reset gradients
    output = model(xs, ys)  # Forward pass
    loss = loss_func(output, ys)  # Compute loss
    loss.backward()  # Backpropagate gradients
    optimizer.step()  # Update model parameters
    return loss.detach().item(), output.detach()  # Return loss and output (detached)

# Generates a set of unique random seeds (used for data generation)
def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

# The main training loop
def train(model, args):
    # Initialize optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    
     # Curriculum controls how n_dims and n_points increase during training
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    
     # Resume from checkpoint if exists
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update() # Sync curriculum to resume point

    # Set up samplers and constants
    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )

    pbar = tqdm(range(starting_step, args.training.train_steps))
    num_training_examples = args.training.num_training_examples

     # Training loop
    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

         # If using sparse tasks, limit valid coordinates for weights
        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        
        # If fixed dataset is specified, sample specific seeds
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # Sample input vectors (xs) from data distribution (e.g., Gaussian)
        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )

        # Create a task instance (e.g., linear regression) and evaluate labels (ys)
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs) # Compute task-specific outputs for xs

        # Choose loss function (e.g., MSE for regression, BCE for classification)
        loss_func = task.get_training_metric()

        # Run a single training step
        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        # Compute pointwise loss for logging
        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        # Compute a baseline loss (e.g., for excess loss normalization)
        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        # Log to Weights & Biases periodically
        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update() # Advance curriculum (more dims/points if scheduled)

        pbar.set_description(f"loss {loss}")

        # Save latest training state (checkpoint)
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        # Optionally save permanent snapshot at defined intervals
        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))

# Initializes model, wandb logging, and calls the train loop
def main(args):
    if args.test_run:
        # For quick testing: fix dims/points and reduce training steps
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        # Initialize Weights & Biases for experiment tracking
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model) # Create Transformer/LSTM model
    model.cuda()
    model.train()

    train(model, args) # Begin training

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # Precompute metrics for eval


# Entry point of the script
if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema) # Load config parser
    args = parser.parse_quinfig() # Parse config file from CLI
    assert args.model.family in ["gpt2", "lstm"] # Sanity check
    print(f"Running with: {args}")

    # Create unique run directory using UUID if not resuming
    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        # Save the config for reproducibility
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args) # Start the process
