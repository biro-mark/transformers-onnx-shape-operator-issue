from transformers import DistilBertForTokenClassification

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.save_pretrained("model")
