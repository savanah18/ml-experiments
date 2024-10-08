# general purpose
import os 
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Literal
import csv

# data processing
import pandas as pd
from rouge_score import rouge_scorer

# configuration
import hydra
from omegaconf import DictConfig

from llama_models.llama3.api.datatypes import UserMessage
from llama_models.llama3.reference_impl.generation import Llama

class OpenRewriteBenchmark(ABC):
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        assert self.data is not None
        assert set(self.data.columns) == set(['source', 'target', 'comment', 'task'])

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def model_healthcheck(self):
        pass

class LlamaOpenRewriteBenchmark(OpenRewriteBenchmark):
    """
    A class used to benchmark text rewriting models.

    ...

    Methods
    -------
    __init__(data_path: str, model_ckpt_dir: str, tokenizer_path: str, model_args: dict = {}) -> None
        Initializes the OpenRewriteBenchmark with the specified paths and model arguments.

    evaluate(data: pd.DataFrame) -> pd.DataFrame
        Evaluates the model's predictions against the target data using the ROUGE-L metric.

    predict(mode: Literal['full', 'sample', 'first_n'], n_rows: int = 1, frac: float = 0.1, prompt_engineered: bool = True) -> pd.DataFrame
        Generates predictions based on the specified mode and returns a DataFrame with the predictions.

    setup(max_seq_len: int = 512, max_batch_size: int = 4, model_parallel_size: Optional[int] = 1) -> None
        Sets up the model with the provided arguments.
    """
    def __init__(
            self, 
            data_path: str,
            model_ckpt_dir: str,
            tokenizer_path: str,
            model_args: dict = {},
        ) -> None:
        """
        Initializes the OpenRewriteBenchmark with the specified paths and model arguments.

        Parameters
        ----------
        data_path : str
            The path to the data file.
        model_ckpt_dir : str
            The directory where the model checkpoints are stored.
        tokenizer_path : str
            The path to the tokenizer.
        model_args : dict, optional
            Additional arguments for the model setup (default is {}).

        Returns
        -------
        None
        """

        super().__init__(data_path)
        self.model_ckpt_dir = model_ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.model_args = dict(model_args)
        self.data = self._parse_text(self.data, 'target')
        self.model = self.setup(**self.model_args['build'])

    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the model's predictions against the target data using the ROUGE-L metric.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the model's predictions and the corresponding target texts.
            It must have two columns: 'prediction' and 'target'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the ROUGE-L precision, recall, and F-measure scores for each prediction.
            The DataFrame has three columns: 'Rouge-L-P', 'Rouge-L-R', and 'Rouge-L-F'.
        """
        print("Evaluating generated output....")
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []
        for index, row in data.iterrows():
            rougeL_score = scorer.score(row['prediction'], row['target'])['rougeL']
            score_dict = {
                'Rouge-L-P': rougeL_score.precision,
                'Rouge-L-R': rougeL_score.recall,
                'Rouge-L-F': rougeL_score.fmeasure
            }
            scores.append(score_dict)
        
        rougel_df = pd.DataFrame(scores, columns=['Rouge-L-P', 'Rouge-L-R', 'Rouge-L-F'])
        # checkpoint
        print("SCORES:\t", rougel_df.mean())
        rougel_df.to_csv('rougel_scores.csv', index=False)
        return rougel_df
    
    def predict(
            self, 
            mode: Literal['full','sample','first_n'], 
            n_rows: int = 1,
            frac: float = 0.1,
            prompt_engineered: bool = True # default to true, meta data is somewhat prompt engineered
        ) -> pd.DataFrame:
        """
        Generates predictions based on the specified mode and returns a DataFrame with the predictions.

        Parameters
        ----------
        mode : Literal['full', 'sample', 'first_n']
            The mode to use for generating predictions. 'full' uses the entire dataset, 'sample' uses a fraction of the dataset, and 'first_n' uses the first n rows.
        n_rows : int, optional
            The number of rows to use if mode is 'first_n' (default is 1).
        frac : float, optional
            The fraction of the dataset to use if mode is 'sample' (default is 0.1).
        prompt_engineered : bool, optional
            Whether to use prompt engineering (default is True).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predictions.
        """
        
        match mode:
            case 'full':
                df = self.data.copy()
            case 'sample':
                df = self.data.sample(frac).copy()
            case 'first_n':
                df = self.data.head(n_rows).copy()
            case _:
                raise ValueError("Invalid mode, Choose from 'full', 'sample', 'first_n'")

        match prompt_engineered:
            case True:
                # little prompt engineering...
                df['prompt'] = "You are provided a piece of text." + \
                    df['comment'] + '. Only include the revised text in the output\n' + \
                    df['source']
            case False:
                df['prompt'] = df['comment'] + '\n\n' + df['source']

        def generate_output(row):
            result = self.model.chat_completion(
                [UserMessage(content=row['prompt'])],
                **self.model_args['generation']
            )
            return result.generation.content
        
        print("Generating output....")
        df['prediction'] = df.apply(generate_output, axis=1)
        df = self._parse_text(df, 'prediction')
        return df

    def setup(
            self,
            max_seq_len: int = 512,
            max_batch_size: int = 4,
            model_parallel_size: Optional[int] = 1,
        ) -> None:

        """
        Sets up the model with the provided arguments.

        Parameters
        ----------
        max_seq_len : int, optional
            The maximum sequence length for the model (default is 512).
        max_batch_size : int, optional
            The maximum batch size for the model (default is 4).
        model_parallel_size : Optional[int], optional
            The size of the model parallelism (default is 1).

        Returns
        -------
        None
        """        
        # TODO make this configuraable
        # environment variables required for torch.distributed.launch
        os.environ['RANK'] = str(0) 
        os.environ['LOCAL_RANK'] = str(1) 
        os.environ['WORLD_SIZE'] = str(1)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
         
        print("Preparing llama model....")
        generator = Llama.build(
            ckpt_dir=self.model_ckpt_dir,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )
        return generator

    def _parse_text(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        # parse target column, remove extra spaces, remove leading and trailing spaces, remove newlines and make it all lowercase
        data[column] = data[column].str.replace('\s+', ' ', regex=True)
        data[column] = data[column].str.strip()
        data[column] = data[column].str.lower()
        data[column] = data[column].str.replace('\n', ' ', regex=True)
        return data

    def model_healthcheck(self, message: Optional[str] = 'Hello llama') -> None:
        """
        Performs a health check on the model by generating a response to a given message.

        Parameters
        ----------
        message : Optional[str], optional
            The message to send to the model for generating a response (default is 'Hello llama').

        Returns
        -------
        None
        """        
        result = self.model.chat_completion(
            [UserMessage(content=message)],
            temperature=0.6,
            top_p=0.9,
            max_gen_len=512,
        )
        out_message = result.generation
        print("*"*25+"\n")
        print(out_message.content)
        print("*"*25+"\n")