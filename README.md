<h1 style="text-align:center;"> ml_template </h1>

A repository template for machine learning projects, following conventions and best practices.

## Structure

-   The `src` layout allows for the easy implementation of a package building pipeline, if ever wanted.
    -   Its inner configuration was engineered to enforce the modularity that PyTorch Lightning standardizes.
-   The `environment.yaml` file allows you to easily track and configure the environment, ideally through Conda.
-   The `conf` directory serves the purpose of organizing `.yaml` configuration files for the Hydra package.
    -   All the subdirectories configure each single module of the package.
-   The `tests` directory has to be used for unit and pipeline testing.
-   The utility files serve the following purposes:
    -   `.gitignore` keeps the repository clean.
    -   `Makefile` simplifies repetitive commands.
    -   `.pre-commit-config.yaml` automatically executes linting and formatting before every commit, keeping code consistent.

# TODOS:

-   [ ] Transform defaults in `default_classifiy` and `default_segment`;
    -   [ ] Add the segmentation defaults;
-   [ ] Improve readme;
