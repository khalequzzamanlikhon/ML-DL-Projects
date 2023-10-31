import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open('saved_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

Wxh = model_data['Wxh']
Whh = model_data['Whh']
Why = model_data['Why']
bh = model_data['bh']
by = model_data['by']
char_to_ix = model_data['char_to_ix']
ix_to_char = model_data['ix_to_char']
vocab_size = len(char_to_ix)  # Define vocab_sizecd

# Function to generate text

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

# Streamlit app
st.title("Text Generation with RNN")

user_input = st.text_area("Enter starting text:", "The sky")

if st.button("Generate"):
    seed = user_input[-1]  # Use the last character of the user input as the seed
    seed_ix = char_to_ix[seed]
    generated_text = sample(np.zeros((100, 1)), seed_ix, 500)
    text_to_print = ''.join(ix_to_char[ix] for ix in generated_text)
    st.write(text_to_print)
