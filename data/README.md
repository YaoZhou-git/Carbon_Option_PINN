# Data access

The numerical single-period and multi-period examples use model inputs and simulated reference outputs. Their supplied checkpoints and reference tables do not require EUA market observations.

The historical estimation uses paired December 2022 and December 2023 EUA futures observations from 2021, accessed through an institutional Refinitiv Datastream subscription. Raw workbooks, daily observed price series, observation-level transformed states, and innovations are not included in this code folder.

See empirical/README.md for the local-input schema and data-directory setting. Obtain the observations through appropriately licensed access and place them outside version control. The environment configuration refers to local files; it does not download data or provide access credentials.

The saved aggregate parameter estimates and model-generated summaries are identified separately from the observations in the empirical documentation.
