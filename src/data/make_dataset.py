import logging
import pandas as pd

class DisasterTweets():
    def __init__(self):
        self.data_path  = "data/raw/"
        
    def load_raw(self):
        self.train = pd.read_csv(f'{self.data_path}train.csv')
        self.test = pd.read_csv(f'{self.data_path}test.csv')
        
        logger.info(f'columns: {self.train.columns.tolist()}')
        logger.info(f'train dataset length: {len(self.train)}, test dataset length: {len(self.test)}')

if __name__=="__main__":  
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
  
    logger=logging.getLogger(__name__)
    ds = DisasterTweets()
    ds.load_raw()