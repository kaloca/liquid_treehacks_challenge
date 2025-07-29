# Liquid AI: TreeHacks Challenges 🌲🌲

## Challenge 2: Operator Optimization

This challenges tests your ability to think cross-functionally across numerics and hardware. You will be optimizing some core computational primitives used in the design space of Liquid AI models, as described in the opening presentation.

Throughout this challenge, [Executorch](https://github.com/pytorch/executorch) will be your friend. We will be using Executorch to transform PyTorch and / or C++ kernels into a representation that can be run on our target device: Samsung Galaxy S24 Ultra.

It could be useful to keep the following resources open as you get started:
* Tutorial on exporting models and operators: [tutorial](https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html)
* Docs on kernel registration: [docs](https://pytorch.org/executorch/stable/kernel-library-custom-aten-kernel.html#custom-ops-api-best-practices)

Any optimization is allowed as long as (a) the results are numerically correct (absolute tolerance of 1e-3 on random inputs) and (b) the custom operator can be run via our profiling [script](pte_android.py) directly on device (one of the Samsung Galaxy S24 Ultra phones provided by the team).

See below for more details on how submissions will be scored.

## Getting Started


Our build system supports both Linux and MacOS (no Windows!).

### Dependencies

The dependencies for this repository could be split into the following groups, don't run anything yet, just read.

1. The git submodules
   * It is mainly the executorch repository + its submodules.
   * `make setup-submodules` pulls the git submodules but does not attempt to sync them or update.
      There are patches applied to the repositories which make the automated management complicated. In any case, you see a problem with a submodule, delete it and re-run that target.
2. Build tools
   * `make check-tools` verifies the basic tools required and suggest how to obtain those
3. Python dependencies
   * `make setup-python-deps` installs the python dependencies declared for that repo, including pytorch, **excluding executorch**.
      The purpose is to have a minimal set of tools for compilation-only needs.
    * Dependencies live in`requirements.txt` and `requirements-torch.txt`
4. Executorch as a python dependency
   * `make setup-executorch` installs executorch and the python dependencies. This calls `install_executorch.sh` (see official documentation). You may need to update your compiler versions and fix your paths.
   * If you are making changes to the executorch repository `./vendor/executorch` you would need to re-run that step to install them.
   * The executorch repo can't be installed as editable.

## One-step build

With your preferred virtual environment (e.g., with conda `conda create -n treehacks python=3.11`), install the dependencies and build the project.

```bash
make setup
```

If `make setup-submodules` fails for you, another option is to clone executorch and move it to `vendor/executorch` before calling `make setup` again.

To customize your executorch build, refer to this [page](https://pytorch.org/executorch/main/getting-started-setup.html). Our setup step will automatically install Executorch with XNNPACK backend enabled (which could be useful to you). If you don't need the backend, you can also install the wheels with pip.

### Running PTEs 

The pte is a representation of your code that can be ingested by a runner. We provide pre-compiled pte runner binaries for Android (`runner/android-arm64-v8a`) and MacOS (`runner/macos-arm64`). You will need these to profile your code (check example usage in `quickstart.py`). These are stored using `git lfs`.

 We also provide pte runner binaries for Linux (`runner/linux`). If you are on a different OS and need to use the pte runner locally, or if you wish to customize it to your needs, you can recompile the Executorch runner following the official instructions.

## Numerical references

We provide a simple mathematical description of the primitives under consideration (`references/math_ref.pdf`). There are multiple approaches to improve your implementation, including purely algorithmic tweaks (e.g., switching from FFT to direct convolution, or Winograd methods) and hardware-specific optimizations (e.g., minimizing overheads, using different data types). Convolutions and recurrences are a cornerstone of numerical computing, and have been implemented and optimized over the years on many different platforms. You can find excellent resources online to get ideas.

## Example custom kernel

See `examples/fftconv` for an end-to-end example of how to register a custom operator with executorch and produce the corresponding pte file. To run the notebook, `python setup.py install` in the ops folder. To export, you will need to adapt to the "out" convention (see kernel registration docs in Executorch), linked above.

### Convolutions

An example of convolution is provided under `examples/direct_conv`. A convolution implementation using FFT and custom kernel registration is provided under `examples/fftconv`.

### Linear recurrences

Some examples using linear scan algorithmsare provided under `examples/linear_scan`.

## Profiling

### Local 

You can gain a lot of insight by measuring the latency and memory usage of your implementation locally (on your laptop, using CPUs). If you want accurate numbers, we also provide phones.

To measure latency on your machine (assuming you have the right pte runner!), use `PYTHONPATH=profiling PTE_RUNNER_PATH=runner/macos-arm64/pte_runner python quickstart.py`. You should see something like this:
```
ProfilingResults(raw=[0.035606, 0.001095, 0.000502, 0.00056], p10=0.0005193999999999999, p50=0.0008275, p90=0.025252700000000003, min=0.000502, avg=0.00944075, max=0.035606)
```
these are the latency measurements.

### Phone 

`adb` is required to run on the phone. See the [docs](https://developer.android.com/tools/adb) and follow the steps. The phones have USB debugging mode already enabled.

To measure on the phone, connect your device to one of the provided Samsung devices and run ` PYTHONPATH=profiling python pte_android.py --pte your_pte_file.pte --runner_path runner/android-arm64-v8a/pte_runner`

### Measurement protocol 

The objective is to minimize latency across pre-specified input shapes, with random inputs (we will measure and average over 100 random inputs, 10 runs, for each setting). 

Target input shapes (batch size, channels, sequence length) for both recurrences and convolutions:
* 1, 512, 64
* 1, 512, 256
* 1, 512, 1024
* 1, 512, 2048
* 1, 512, 4096
* 1, 512, 16384
* 1, 512, 65536

* 1, 2048, 64
* 1, 2048, 256
* 1, 2048, 1024
* 1, 2048, 2048
* 1, 2048, 4096
* 1, 2048, 16384
* 1, 2048, 65536

In addition, for convolutions, we will measure filter sizes: 4, 32, 128, and as long as the input sequence (max 65536), resulting in 4 times the number of measurements (we will still average across filter size settings). You can provide a pte that accepts dynamic shapes, or different ptes for each shape, allowing you to optimize for the best performance.

We will check for numerical correctness with absolute tolerance of 1e-3 on random and structured inputs (e.g., a tensor of ones). 

### Issues

#### Issues with data type support during compilation

These are generally resolved by making sure `gcc` is up-to-date, or switching to a different toolchain, for example `llvm` on arm64.


#### Exec format error

`OSError: [Errno 8] Exec format error`

This is generally due to a mismatch between the pte runner and the architecture of the machine you are running on. To measure latency locally, you can use compile and use Executorch's native runner.




