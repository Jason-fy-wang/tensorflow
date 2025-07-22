from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
#distilbert/distilbert-base-uncased-finetuned-sst-2-english
model_path = os.path.join(os.getcwd(), "model","distilbert-base-uncased-finetuned-sst-2-english")


model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_path)

texts = ["i love program", "i hate program"]


for text in texts:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**tokens).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        #print(f"probes: {probs}, pred :{pred}")
        confidence = probs[0][pred]
        labels = ["negative", "postive"]
        print(f"Text: {text}\n sentiments: {labels[pred]} with confidence {confidence}")


