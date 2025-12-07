# How to Contribute

We welcome contributions to BERT-CLAM! This guide will help you get started with contributing to the project.

## Adding a New Strategy

Adding a new continual learning strategy is one of the best ways to contribute to BERT-CLAM. The framework's modular design makes it straightforward to implement and integrate new strategies.

![Adding a New Strategy Flowchart](../../assets/diagrams/add_new_strategy.png)

### Implementation Steps

1. **Create Strategy Class**: Inherit from `ContinualLearningStrategy` and implement the `apply()` method
2. **Register in Factory**: Add your strategy type to the strategy factory in `run_experiment.py`
3. **Update Model**: Integrate your strategy module into `BERTCLAMModel`
4. **Configure**: Create a JSON configuration file for your strategy
5. **Test**: Run experiments to validate your implementation

For detailed implementation examples, see the [Architecture documentation](../concepts/architecture.md) and existing strategy implementations in the codebase.

## Other Ways to Contribute

- Report bugs and issues
- Improve documentation
- Add unit tests
- Optimize performance
- Share experiment results

## Getting Help

If you have questions or need assistance, please open an issue on GitHub or reach out to the maintainers.