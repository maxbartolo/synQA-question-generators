# synQA-question-generators

Question generation models used in the work "[Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation](https://arxiv.org/abs/2104.08678)" and expanded on in "[Models in the Loop: Aiding Crowdworkers with Generative Annotation Assistants](https://arxiv.org/abs/2112.09062)".

## Models

There are 3 models available:
* `generator_qa_squad`: trained on the 87k [SQuADv1.1](https://rajpurkar.github.io/SQuAD-explorer/) training examples.
* `generator_qa_adversarialqa`: trained on the 30k combined [AdversarialQA](https://adversarialqa.github.io/) training examples.
* `generator_qa_squad_plus_adversarialqa`: trained on the 30k combined AdversarialQA training examples and 10k randomly selected SQuAD examples.

For more information on the generative models, see [Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation](https://arxiv.org/abs/2104.08678).

If you are just looking for data generated using these models, see https://github.com/maxbartolo/improving-qa-model-robustness.

## Installation Instructions

```bash
# Clone the repo
git clone https://github.com/maxbartolo/synQA-question-generators
cd synQA-question-generators

# Set up a conda environment
conda create -n synQA python=3.8
conda activate synQA
pip install -r requirements.txt

# Download the models
python download_models.py
```

## Example usage:

To use a QA generation model, use:

```bash
python generate.py
```

The script can optionally take the following arguments:
```
model: The model you want to use to generate questions
--context: The context you want the model to generate a question for
--answer: The answer you want to condition the model on. Must be a span from the context
--num_to_generate: The number of examples to generate
```

An more advanced example is:

```bash
python generate.py "generator_qa_adversarialqa" --answer "Seattle" --context "The DADC workshop will be held at NAACL '22 in Seattle. Seattle (/siˈætəl/ is a seaport city on the West Coast of the United States. With a 2020 population of 737,015, it is the largest city in both the state of Washington and the Pacific Northwest region of North America. The Seattle metropolitan area's population is 4.02 million, making it the 15th-largest in the United States. Its growth rate of 21.1% between 2010 and 2020 makes it one of the nation's fastest-growing large cities." --num_to_generate 5
```
