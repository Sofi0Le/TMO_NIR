import spacy
from spacy.tokens import Span
from spacy.training import Example

# Создаем новый тип именованной сущности
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label("TOPIC")
ner.add_label("PRICE")

# Подготовим набор данных для обучения
TRAINING_DATA = [
    ("машины за 200", {"entities": [(0, 6, "TOPIC"), (10, 13, "PRICE")]}),
    ("машины 200", {"entities": [(0, 6, "TOPIC"), (8, 11, "PRICE")]}),
    ("тема машины стоимость 200", {"entities": [(5, 11, "TOPIC"), (22, 25, "PRICE")]}),
    ("песни за 200", {"entities": [(0, 5, "TOPIC"), (9, 12, "PRICE")]}),
    ("песни 500", {"entities": [(0, 5, "TOPIC"), (6, 9, "PRICE")]}),
    ("тема страны стоимость 1000", {"entities": [(5, 11, "TOPIC"), (22, 26, "PRICE")]}),
    ("страны стоимость 1000", {"entities": [(0, 6, "TOPIC"), (17, 21, "PRICE")]}),
]

# Начнем обучение
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):  
    optimizer = nlp.begin_training()
    for itn in range(100):
        losses = {}
        for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print(losses)

# Сохраним модель
nlp.to_disk("model")
