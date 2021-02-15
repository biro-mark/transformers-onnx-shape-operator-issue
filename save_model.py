from transformers import DistilBertForTokenClassification

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.save_pretrained("model")

python -m transformers.convert_graph_to_onnx --model ../model --framework pt --tokenizer distilbert-base-uncased out.onnx
