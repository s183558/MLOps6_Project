import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

logger=logging.getLogger(__name__)

def processing(df):
    df = df.fillna('No info', axis=1)
    df['input'] = 'KEYWORD: ' + df.keyword + '; LOCATION: ' + df.location + '; TEXT: ' + df.text
    df['input'] = df['input'].str.lower()
    df = df.drop(columns=['keyword','location','text'], axis=1)
    df.fillna("No info", inplace=True)
    df.drop_duplicates(inplace=True)
    return df

class DisasterTweets():
    def __init__(self):
        self.data_path  = "data/raw/"
        self.load_raw()
        self.process_datasets()

    def load_raw(self):
        self.train = pd.read_csv(f'{self.data_path}train.csv')
        self.test = pd.read_csv(f'{self.data_path}test.csv')
        
        logger.info(f'train columns: {self.train.columns.tolist()}')
        logger.info(f'test columns: {self.test.columns.tolist()}')
        logger.info(f'train dataset length: {len(self.train)}, test dataset length: {len(self.test)}')
        
    def process_datasets(self):
        self.train = processing(self.train)
        self.train['target'] = np.float64(self.train['target'])
        self.test = processing(self.test)


if __name__=="__main__":  
    ds = DisasterTweets()

    