class Vocabulary(object) :
  def __init__(self,token_to_idx=None,add_unk=True,unk_token="<UNK>") :
     if token_to_idx is None :
        token_to_idx = {}
     self._token_to_idx= token_to_idx
     self._idx_to_token = {idx: token for token,idx in self._token_to_idx.items() }
     
     self._add_unk = add_unk
     self._unk_token = unk_token
     
     self.unk_index = -1
     
     if add_unk :
        self.unk_index = self.add_token(unk_token)
        

  def add_token(self,token) :
      if token in self._token_to_idx :
         index = self._token_to_idx[token]
      else:
         index = len(self._token_to_idx)
         self._token_to_idx[token] = index
         self._idx_to_token[index] = token
      return index
      
  def lookup_token(self,token) :
      if self.unk_index >= 0 :
         return self._token_to_idx.get(token,self.unk_index)
      else  :
         return self._token_to_idx[token]
         
         
  def to_serializable(self) :
      return {'token_to_idx' : self._token_to_idx,  
              'add_unk' : self._add_unk,
              'unk_token' : self._unk_token
              }
              
  def __len__(self) :
     return len(self._token_to_idx)  
          
   