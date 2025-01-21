# DeepQRank: Generating Search Engine Rankings with Deep Reinforcement Learning

## Overview

**DeepQRank** is an innovative approach to the Learning to Rank (LTR) problem, utilizing deep Q-learning to generate search engine rankings. This project demonstrates how reinforcement learning can be applied to improve the relevance of document rankings in response to user queries, offering a state-of-the-art performance in ranking tasks.

### Key Features:
- **Deep Q-Learning**: Utilizes deep reinforcement learning to train an agent for sequential decision making in ranking documents.
- **Flexibility**: Capable of handling both pointwise and listwise ranking, unlike traditional supervised learning methods which are limited to pointwise ranking.
- **Performance**: Achieves an NDCG@1 score of 0.5075 on Microsoft's LETOR dataset, outperforming leading supervised learning models like SVMRank.

## Project Structure

- **scripts/**: Contains scripts for training and evaluating the model.
  - `train.py`: Script to train the DeepQRank model.
  - `evaluate.py`: Script to evaluate the model's performance.
- **model/**: Includes the model architecture.
  - `dqn.py`: Defines the Deep Q-Network used for ranking.
- **data/**: Placeholder for datasets (e.g., LETOR listwise dataset).
- **docs/**: Documentation and research papers related to the project.
- **README.md**: This file.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository:**
   ```
   git clone https://github.com/abhi1345/deep-q-rank.git
   cd deep-q-rank
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

   Ensure you have the following libraries installed:
   - PyTorch
   - NumPy
   - Pandas
   - SciPy
   - Scikit-learn

## Usage

### Training the Model

To train the DeepQRank model:

```
python scripts/train.py --data_path data/letor --epochs 100 --batch_size 32
```

### Evaluating the Model

After training, evaluate the model's performance:

```
python scripts/evaluate.py --model_path model/dqn.pth --data_path data/letor
```

### Modifying the Neural Network

You can modify the neural network architecture by editing `model/dqn.py`. This file contains the implementation of the Deep Q-Network used for ranking.

## Performance Metrics

**Normalized Discounted Cumulative Gain (NDCG)** at position 1 (NDCG@1) is used to evaluate the model's performance:

| Model         | NDCG@1 |
|---------------|--------|
| RankSVM       | 0.4958 |
| ListNet       | 0.4002 |
| AdaRank-MAP   | 0.3821 |
| AdaRank-NDCG  | 0.3876 |
| SVMMAP        | 0.3853 |
| RankNet       | 0.4790 |
| MDPRank       | 0.4061 |
| **DeepQRank** | **0.5075** |

## Contributing

Contributions are welcome! Here's how you can contribute:

- **Report Bugs**: Open an issue on GitHub.
- **Suggest Features**: Discuss new features or improvements in the issues section.
- **Submit Pull Requests**: Enhance the codebase with your contributions.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to Microsoft for providing the LETOR dataset.
- Special thanks to the research community for their contributions to the field of Learning to Rank and Deep Reinforcement Learning.

## Contact

For any inquiries or collaboration, please contact [Abhishek Sharma](mailto:your-email@example.com).

---

Feel free to explore, fork, and contribute to this project to help advance the field of search engine optimization through deep reinforcement learning!
