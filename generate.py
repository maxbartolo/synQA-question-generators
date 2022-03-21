import argparse
import logging
import os
import time
import warnings
from fairseq.models.transformer import TransformerModel


"""
Example usage:
python generate.py "generator_qa_adversarialqa" --answer "Seattle" --context "The DADC workshop will be held at NAACL '22 in Seattle. Seattle (/siˈætəl/ is a seaport city on the West Coast of the United States. With a 2020 population of 737,015, it is the largest city in both the state of Washington and the Pacific Northwest region of North America. The Seattle metropolitan area's population is 4.02 million, making it the 15th-largest in the United States. Its growth rate of 21.1% between 2010 and 2020 makes it one of the nation's fastest-growing large cities." --num_to_generate 5
"""

# Set up logging
logging.basicConfig(level=logging.DEBUG)


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_NAMES = [
    'generator_qa_squad', 
    'generator_qa_adversarialqa', 
    'generator_qa_squad_plus_adversarialqa'
]
model_paths = {model_name: os.path.join(MODELS_DIR, model_name) for model_name in MODEL_NAMES}

SPECIAL_TOKENS = {
    'bos_token': '<s>',
    'eos_token': '</s>',
    'sep_token': '</s>'
}


def convert_example_to_input(example):
    ex_input_inner = f" {SPECIAL_TOKENS['sep_token']} ".join(example)
    ex_input = (
        f"{SPECIAL_TOKENS['bos_token']} {ex_input_inner} {SPECIAL_TOKENS['eos_token']}"
    )
    return ex_input


def clean_special_tokens(text):
    for _, special_tok in SPECIAL_TOKENS.items():
        text = text.replace(special_tok, "")
    return text.strip()


def main(args):
    if not args.context:
        raise BaseException("A contexts is required.")

    if args.answer not in args.context:
        warnings.warn(f"The answer provided ({args.answer}) is not in the context.") 

    assert args.model in model_paths, f"Model ({args.model}) has not yet been implemented. The available options are: {'|'.join(MODEL_NAMES)}"
    model_path = model_paths[args.model]

    # Load the model
    logging.info(f"Loading model from {model_path}")
    generator = TransformerModel.from_pretrained(
        model_path,
        checkpoint_file='checkpoint_best.pt',
        bpe='gpt2',
        fp16=True,
    )
    logging.info(f"Model ({args.model}) loaded")

    decode_params = {
        'beam': 10, 
        'sampling': True, 
        'sampling_topp': 0.9
    }

    example = [args.answer, args.context]
    ex_input = convert_example_to_input(example)
    ex_inputs = [ex_input]

    # Generate some questions
    for _ in range(args.num_to_generate):
        t_0 = time.time()
        output = generator.translate(ex_inputs, **decode_params)

        if isinstance(output, str):
            clean_output = clean_special_tokens(output)
        else:
            clean_output = [clean_special_tokens(q) for q in output]
            if len(clean_output) == 1:
                clean_output = clean_output[0]
        
        logging.info(f"Generated: {clean_output} | Time taken: {time.time() - t_0:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help=f"The model you want to use to generate questions. The available options are: {'|'.join(MODEL_NAMES)}.", type=str)
    parser.add_argument("--context", help="The context you want the model to generate a question for.", type=str, required=True)
    parser.add_argument("--answer", help="The answer you want to condition the model on. Must be a span from the context.", type=str)
    parser.add_argument("--num_to_generate", help="The number of examples to generate", type=int, default=5)
    args = parser.parse_args()

    main(args)
