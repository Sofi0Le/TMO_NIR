import spacy

# Загрузим обученную модель
nlp = spacy.load("model")

# Функция извлечения темы и стоимости
def extract_topic_and_price(text):
    doc = nlp(text)
    topic = None
    price = None
    for ent in doc.ents:
        if ent.label_ == "TOPIC":
            topic = ent.text
        elif ent.label_ == "PRICE":
            price = ent.text
    return topic, price

# Пример использования функции

while True:
    text = input('Введите темы и стоимость: ')
    if text == "стоп":
        break

    topic, price = extract_topic_and_price(text)
    print("Тема:", topic)
    print("Стоимость:", price)
