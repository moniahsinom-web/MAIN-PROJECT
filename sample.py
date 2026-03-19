from transformers import pipeline
 
classifier = pipeline("sentiment-analysis")
 
review = "Product Quality Is Very Good"
result = classifier(review)
 
print(result)