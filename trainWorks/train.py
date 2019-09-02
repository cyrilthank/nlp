from general import *
from ReviewDataset import *
from ReviewClassifier import ReviewClassifier

print(args.review_csv)

if args.reload_from_files :
  print("loading dataset and vectorizer")
  dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,args.vectorizer_file)
else :
  print("loading dataset and making vectorizer")
  dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
  dataset.save_vectorizer(args.vectorizer_file)

vectorizer=dataset.get_vectorizer()

classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))

classifier = classifier.to(args.device)

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(),lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',factor=0.5,patience=1)

dataset.set_split('train')

dataset.set_split('val')

try:

   for epoch_index in range(args.num_epochs) :
      dataset.set_split('train')
      batch_generator = generate_batches(dataset,batch_size=args.batch_size,device=args.device)
      
      running_loss=0.0
      running_acc=0.0
      
      classifier.train()
      
      for batch_index,batch_dict in enumerate(batch_generator) :
          #Step 1 . zero the grad
          optimizer.zero_grad()
          
          #Step 2 . compute the output
          y_pred = classifier(x_in=batch_dict['x_data'].float())
          
          #Step 3 compute the loss_func
          loss = loss_func(y_pred,batch_dict['y_target'].float())
          loss_t = loss.item()
          running_loss += (loss_t - running_loss)/(batch_index + 1)
          
          #Step 4 backprop
          loss.backward()
          
          #Step 5 update the gradient
          optimizer.step()
          

except KeyboardInterrupt:
   print("Exiting Loop")          