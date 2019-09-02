
import json
import pandas as pd
from torch.utils.data import Dataset,DataLoader

from ReviewVectorizer import ReviewVectorizer

class ReviewDataset(Dataset) :
  def __init__(self,review_df,vectorizer) :
     self.review_df = review_df
     self._vectorizer = vectorizer
     
     self.train_df = self.review_df[self.review_df['split']=='train']
     self.train_size = len(self.train_df)
     
     self.val_df = self.review_df[self.review_df['split']=='val']
     self.validation_size = len(self.val_df)
     
     self.test_df = self.review_df[self.review_df['split']=='test']
     self.test_size = len(self.test_df)
     
     self._lookup_dict = {
                          'train' : (self.train_df,self.train_size),
                          'val'   : (self.val_df,self.validation_size),
                          'test'  : (self.test_df,self.test_size)}
                          
     self.set_split('train')
     
  @classmethod   
  def load_dataset_and_make_vectorizer(cls,review_csv) :
      review_df = pd.read_csv(review_csv)
      train_review_df = review_df[review_df.split=='train']
      return cls(review_df,ReviewVectorizer.from_dataframe(train_review_df))   

  def load_dataset_and_load_vectorizer(cls,review_csv,vectorizer_filepath) :
      review_df = pd.read_csv(review_csv)
      vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
      return cls(review_df,vectorizer)
      
  @staticmethod
  def load_vectorizer_only(vectorizer_filepath) :
      with open(vectorizer_filepath) as fp :
           return ReviewVectorizer.from_serializable(json.load(fp));  

  def save_vectorizer(self,vectorizer_filepath) :
      with open(vectorizer_filepath,"w") as fp :
           json.dump(self._vectorizer.to_serializable(), fp)
               
              
  
  def get_vectorizer(self) :
     return self._vectorizer
     
  def set_split(self,split="train") :
      self._target_split = split
      self._target_df,self._target_size = self._lookup_dict[split]
      
  def __len__(self) :
       return self._target_size
       
       
  def __getitem__(self,index) :
      row= self._target_df.iloc[index]
       
      review_vector = self._vectorizer.vectorize(row.review)
      
      rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)
      
      return {
             'x_data' : review_vector,
             'y_target' : rating_index
             }
             
             
  def get_num_batches (self,batch_size) :
      return len(self)//batch_size
      

def generate_batches(dataset,batch_size,shuffle=True,drop_last=True,device="cpu")   :
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)
    
    for data_dict in dataloader :
       out_data_dict={}
       for name,tensor in data_dict.items() :
          out_data_dict[name] = data_dict[name].to(device)
          
       yield out_data_dict
       
    
'''
def load_dataset_make_vectorizer(review_csv) :
   review_df = pd.read_csv(review_csv)
   train_review_df = review_df[review_df.split=='train']
   print(train_review_df)
'''   