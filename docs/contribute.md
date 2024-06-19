# Contributing to PhysioEx

Welcome to the PhysioEx project! We appreciate your interest in contributing. 

## Ways to contribute

Numerous avenues exist for contributing to the PhysioEx project, with the most prevalent being the provision of code or enhancements to the documentation. The refinement of the documentation is equally as vital as the augmentation of the library itself.

Should you discover an error in the documentation, or have implemented improvements, we encourage you to post a new message on the GitHub discussion board or, ideally, submit a GitHub pull request.

An additional method of contribution involves reporting any issues you encounter, and endorsing issues reported by others that are pertinent to you by giving them a “thumbs up”. Your support in promoting the project is also beneficial: mention the project in your blog posts and articles, provide a link to it from your website, or simply give it a star to indicate your usage of it.

### Adviced contributing fields

1. Adding support for a new physiological signal dataset (e.g. checking [PhysioNet](https://physionet.org) or [MOABB](https://moabb.neurotechx.com/docs/))
2. Adding support for a new deep learning architecture for physiological signal analysis.
3. Proposing a novel Explainable AI algorithm.

Consider also this other very important fields of contribution:

- Documentation improving, note: check the official [MkDocs Material doc](https://squidfunk.github.io/mkdocs-material/). as reference.
- Code improving and issues solving.

## How to Contribute

We welcome contributions from everyone. PhysioEx is a GitHub hosted libray, to know more about GitHub collaborative devolpment check their [ufficial doc](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models).To get started, follow these steps:

1. Fork the Repository

Fork this repository to your GitHub account by clicking on the 'Fork' button on the top right of the page. This creates a copy of the repository in your account.

2. Clone your fork of the physioex repo from your GitHub account to your local disk:

```bash
    $ git clone git@github.com:your-username/physioex.git 
    $ cd physioex 
```

Replace `your-username` with your GitHub username.

4. Make sure to have anaconda or miniconda correctly installed in your machine, then start installing a new virtual enviroment
```bash
    $ conda create -n myenv python==3.10
```    

5. Now jump into the enviroment and upgrade pip
```bash
    $ conda activate myenv
    $ conda install pip
    $ pip install --upgrade pip
```

6. To install PhysioEx in development mode run:
```bash
    $ pip install -e .
```    

7. Now you need to keep your fork and the original physioex repo in sync creating an upstream:
```bash
    $ git remote add upstream https://github.com/guidogagl/physioex.git
```    

!!! warning Check that everything went allright
    To check that the upstream is correctly setted up, run:
    ```bash
        $ git remote -v
    ```
    And check that the output resembles:
    ```bash
        > origin    https://github.com/your-username/physioex.git (fetch)

        > origin    https://github.com/your-username/physioex.git (push)

        > upstream  https://github.com/guidogagl/physioex.git (fetch)
        > upstream  https://github.com/guidogagl/physioex.git (push)
    ```

!!! tip Keep your fork updated
    To keep your fork repo updated check the [official doc](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)

8. Write your code & documentation and when it's ready submit a Pull Request! For a step-by-step guide on how to submit a PR check the [GitHub official documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)