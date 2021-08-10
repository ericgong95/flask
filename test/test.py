import requests

# Request Fails Validation
r1 = requests.get('http://localhost:5000/embeddings?test=the+quick+brown+fox')
print(r1.content)

# Request Passes
r2 = requests.get('http://localhost:5000/embeddings?sentence=the+quick+brown+fox')
print(r2.content)

# Request Fails Validation
base_data = {
    "test": [
        "the quick brown fox jumped over the lazy dog",
        "the five boxing wizards jump quickly"
    ]
}
r3 = requests.post('http://localhost:5000/embeddings/bulk',json=base_data)
print(r3.content)


# Request Passes
base_data = {
    "sentences": [
        "the quick brown fox jumped over the lazy dog",
        "the five boxing wizards jump quickly"
    ]
}
r4 = requests.post('http://localhost:5000/embeddings/bulk',json=base_data)
print(r4.content)

# Request Fails Validation
base_data = {
  "sentence_1": "the quick brown fox jumped over the lazy dog",
  "sentence_2":"the five boxing wizards jump quickly",
  "sentence_3": "test"
}
r5 = requests.post('http://localhost:5000/embeddings/similarity',json=base_data)
print(r5.content)

# Request Passes
base_data = {
  "sentence_1": "the quick brown fox jumped over the lazy dog",
  "sentence_2":"the five boxing wizards jump quickly"
}
r6 = requests.post('http://localhost:5000/embeddings/similarity',json=base_data)
print(r6.content)