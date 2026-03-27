import torch

class RAGModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = openai.Completion.create(model='text-davinci-003').model

    def generate(self, prompt):
        output = self.model.generate(prompt=prompt)
        return output.choices[0].text