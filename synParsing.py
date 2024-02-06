import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import spacy
from collections import defaultdict
import nltk
from nltk.corpus import opinion_lexicon

nltk.download('opinion_lexicon')

# Load the English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')


def plot_metrics(results):
    # Prepare the data
    rules = list(results.keys())
    precisions = [results[rule]['precision'] for rule in rules]
    recalls = [results[rule]['recall'] for rule in rules]

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Set bar width
    bar_width = 0.35

    # Positions of the left bar-boundaries
    bar_l = [i+1 for i in range(len(rules))]

    # Positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    # Create precision bars
    ax.bar(bar_l,
           precisions,
           width=bar_width,
           color='b',
           align='center',
           label='Precision')

    # Create recall bars
    ax.bar(bar_l,
           recalls,
           width=bar_width,
           color='r',
           align='edge',
           label='Recall')

    # Set the ticks to be rule names
    plt.xticks(tick_pos, rules, rotation=45)

    # Set the y-axis label
    ax.set_ylabel('Score')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


def parse_xml(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()

    data = []
    for sentence in root.findall('sentence'):
        sentence_id = sentence.get('id')
        text = sentence.find('text').text

        aspect_terms = []
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is not None:
            for aspectTerm in aspectTerms.findall('aspectTerm'):
                term = aspectTerm.get('term')
                polarity = aspectTerm.get('polarity')
                aspect_terms.append((term, polarity))
        data.append((sentence_id, text, aspect_terms))
    return data


positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())
NEGATION_ADVERBS = {'not', 'never', 'no', 'nobody',
                    'none', 'nowhere', 'nothing', 'neither', 'no one'}
POSITIVE_CONJUNCTIONS = {'and', 'also', 'moreover'}
NEGATIVE_CONJUNCTIONS = {'but', 'however', 'although'}
NEUTRAL_CONJUNCTIONS = {'while', 'whereas'}


def absa(sentence, aspect_term):
    doc = nlp(sentence)

    sentiment = "neutral"
    negation = False
    rule_used = None

    for token in doc:
        if token.text == aspect_term:
            for child in token.children:

                # Rule 1: Direct Word Sentiment
                if child.text in positive_words or child.lemma_ in positive_words:
                    sentiment = 'positive'
                    rule_used = 'direct_word_positive'

                elif child.text in negative_words or child.lemma_ in negative_words:
                    sentiment = 'negative'
                    rule_used = 'direct_word_negative'

                # Rule 2: Adjective Modifier Sentiment
                if child.dep_ == 'amod':
                    if child.text in positive_words or child.lemma_ in positive_words:
                        sentiment = 'positive'
                        rule_used = 'adjective_modifier_positive'

                    elif child.text in negative_words or child.lemma_ in negative_words:
                        sentiment = 'negative'
                        rule_used = 'adjective_modifier_negative'

                    else:
                        sentiment = 'neutral'
                        rule_used = 'adjective_modifier_neutral'

                # Rule 3: Adverb Sentiment
                if child.pos_ == 'ADV':
                    if child.text in positive_words or child.lemma_ in positive_words:
                        sentiment = 'positive'
                        rule_used = 'adverb_positive'

                    elif child.text in negative_words or child.lemma_ in negative_words:
                        sentiment = 'negative'
                        rule_used = 'adverb_negative'

                    else:
                        sentiment = 'neutral'
                        rule_used = 'adverb_neutral'

                # Negation rule
                if child.dep_ == 'neg' or child.text in NEGATION_ADVERBS:
                    negation = not negation
                    rule_used = 'negation'

                # # Conjunction rules
                # elif child.pos_ == 'CONJ':
                #     if child.lemma_ in POSITIVE_CONJUNCTIONS:
                #         sentiment = 'positive'
                #         rule_used = 'positive_conjunction'
                #     elif child.lemma_ in NEGATIVE_CONJUNCTIONS:
                #         sentiment = 'negative'
                #         rule_used = 'negative_conjunction'
                #     elif child.lemma_ in NEUTRAL_CONJUNCTIONS:
                #         sentiment = 'neutral'
                #         rule_used = 'neutral_conjunction'

    # Apply negation rule
    if negation:
        sentiment = 'negative' if sentiment == 'positive' else 'positive'
        rule_used = 'negation'

    return sentiment, rule_used


# To calculate precision and recall
def calculate_precision_recall(predictions, ground_truth):
    tp = 0
    fp = 0
    fn = 0

    for aspect, sentiment in predictions.items():
        if sentiment == ground_truth[aspect]:
            tp += 1
        else:
            if sentiment != 'neutral':
                fp += 1
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def main():
    # Load data
    data = parse_xml('Restaurants.xml')

    # Keep track of predictions and ground truth for each rule
    results = defaultdict(
        lambda: {"predictions": {}, "ground_truth": {}, "precision": 0, "recall": 0})

    # Get aspect terms and polarity for each sentence
    for sentence_id, text, aspect_terms in data:
        for term, polarity in aspect_terms:
            sentiment, rule = absa(text, term)
            # Store the sentence along with the aspect term in the dictionary
            results[rule]["predictions"][(term, text)] = sentiment
            results[rule]["ground_truth"][(term, text)] = polarity

    # Calculate precision and recall for each rule
    for rule, result in results.items():
        predictions = result["predictions"]
        ground_truth = result["ground_truth"]
        precision, recall = calculate_precision_recall(
            predictions, ground_truth)
        results[rule]["precision"] = precision
        results[rule]["recall"] = recall

        print(f"Rule: {rule}")
        print("Precision:", precision)
        print("Recall:", recall)
        print("Successful Cases:")
        successful_cases = [(aspect, sentence) for (aspect, sentence), sentiment in predictions.items(
        ) if sentiment == ground_truth[(aspect, sentence)]]
        if successful_cases:
            for aspect, sentence in successful_cases[:1]:
                print(
                    f"- Aspect: {aspect}, Sentiment: {predictions[(aspect, sentence)]}, Ground Truth: {ground_truth[(aspect, sentence)]}, Sentence: {sentence}")
        else:
            print("No successful cases.")
        print("Failure Cases:")
        failure_cases = [(aspect, sentence) for (aspect, sentence), sentiment in predictions.items(
        ) if sentiment != ground_truth[(aspect, sentence)]]
        if failure_cases:
            for aspect, sentence in failure_cases[:1]:
                print(
                    f"- Aspect: {aspect}, Sentiment: {predictions[(aspect, sentence)]}, Ground Truth: {ground_truth[(aspect, sentence)]}, Sentence: {sentence}")
        else:
            print("No failure cases.")
        print()

    plot_metrics(results)


if __name__ == "__main__":
    main()
