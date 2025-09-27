#NanoGPT

For my Seasons of Code(2025) project, I modified Andrej Karpathy's NanoGPT to function as a conversational chatbot trained on a dataset from kaggle.

It is a Transformer-based neural network designed to generate text, character by character or token by token. It uses causal self-attention, which ensures the model only looks at past tokens when generating the next one — this is what makes it good for generating text in a left-to-right way.
Due to GPU limitations, the ouput is not very accurate, but with a more powerful GPU's and hyperparameter optimization the output can be improved.
