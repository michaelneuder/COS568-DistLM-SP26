## Assignment 2: Distributed Training of Language Models

Mike Neuder, COS 568, Spring 2026

### Task 1

The first task only required small code changes to `run_glue.py`. 

- The model is loaded from hugging face using the arg passed in the CLI flags
```
model = model_class.from_pretrained(args.model_name_or_path, config=config)
```
- The backward pass is the following method on the loss object
```
loss.backward()
```
- The optimizer step can be called directly
```
optimizer.step()
```
- The evaluation method is simply
```
evaluate(args, model, tokenizer)
```

The loss of the first five minibatches is:

```
step 0, loss 0.7691709399223328
step 1, loss 0.7817339301109314
step 2, loss 0.6885837912559509
step 3, loss 0.7662752866744995
step 4, loss 0.7341869473457336
```

And the per-epoch accuracies for the three epochs of training are 

```
03/06/2026 19:26:48 - INFO - __main__ -     acc = 0.628158844765343
03/06/2026 19:30:57 - INFO - __main__ -     acc = 0.6498194945848376
03/06/2026 19:35:09 - INFO - __main__ -     acc = 0.6209386281588448
```

### Task 2

#### Part A

In task 2, we run for a single epoch with the following extra CLI flags

```
  --master_ip  10.10.1.2 \
  --master_port 12345 \
  --world_size 4 \
  --local_rank 0
```

where we change the local rank based on the node we are running on. We also modfiy the batch size to 16 examples per minibatch per machine
```
--per_device_train_batch_size 16
```

Further, we need to use `DistributedSampler` to partition the data into groups of 16. I added a timing loop that measures the iteration time of each except the first minibatch with the following code.

```
if step > 0:
    iter_times.append(time.time() - iter_start)
```
and get an average time over the 38 iterations of:

```
Average time per iteration: 24.781s (38 iterations)
```

Further, I collect the loss at each step for each node and plot it on a 2x2 grid:

![task2a2b.png](task2a2b.png)

Notice that since we used the same seed and configuration, the losses are identical between both runs. 

#### Part B

The code here is much simpler than the manual scatter gather in Part A. The builtin function of `all_reduce` allows us to implement the gradient averaging across the nodes in the following lines of code:

```
for p in model.parameters():
    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
    p.grad /= args.world_size
```

With a much shorter average runtime:
```
Average time per iteration: 9.404s (38 iterations)
```

The loss is shown in the figure above to be exactly the same as in Part A. 

### Task 3

This version of the code simply registers the model as DistributedDataParallel with the following code:

```
if args.local_rank != -1:
    model = DistributedDataParallel(model)
```

with an average iteration time 

```
Average time per iteration: 7.877s (38 iterations)
```

The loss is shown below with the other tasks

![./task3.png](./task3.png)

Clearly the loss is taking the same shape, but isn't numerically exactly the same as Task 2A/2B. We will discuss this more in the next section, but I believe this is due to the asynchronous nature of DistributedDataParallel, which may only approximate the gradient sum because it overlaps the communication with the backwards pass. This is described in Section 3.2.3 of https://arxiv.org/pdf/2006.15704, 

> Hence, the reverse order should approximately
represent the gradient computation order in the backward
pass

which tells us that the gradient updates are only approximately correct, but numerically very similar to the sequential all_reduce operation. 

### Task 4
