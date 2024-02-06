import pandas as pd
import plotly.graph_objects as go

data = [
    {
        "Rule": "None",
        "Precision": 1.0,
        "Recall": 0.19196119196119196,
        "Successful Cases": "- Aspect: menu, Sentiment: neutral, Ground Truth: neutral, Sentence: The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
        "Failure Cases": "- Aspect: staff, Sentiment: neutral, Ground Truth: negative, Sentence: But the staff was so horrible to us."
    },
    {
        "Rule": "adjective_modifier_positive",
        "Precision": 0.8958904109589041,
        "Recall": 0.8958904109589041,
        "Successful Cases": "- Aspect: kitchen, Sentiment: positive, Ground Truth: positive, Sentence: The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not.",
        "Failure Cases": "- Aspect: scents, Sentiment: positive, Ground Truth: negative, Sentence: The strong scents coming from the left and right of me negatively affected my taste buds."
    },
    {
        "Rule": "adjective_modifier_neutral",
        "Precision": 1.0,
        "Recall": 0.16981132075471697,
        "Successful Cases": "- Aspect: cheese, Sentiment: neutral, Ground Truth: neutral, Sentence: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.",
        "Failure Cases": "- Aspect: perks, Sentiment: neutral, Ground Truth: positive, Sentence: Not only was the food outstanding, but the little 'perks' were great."
    },
    {
        "Rule": "adjective_modifier_negative",
        "Precision": 0.6530612244897959,
        "Recall": 0.6530612244897959,
        "Successful Cases": "- Aspect: service, Sentiment: negative, Ground Truth: negative, Sentence: From the terrible service, to the bland food, not to mention the unaccommodating managers, the overall experience was horrible.",
        "Failure Cases": "- Aspect: atmosphere, Sentiment: negative, Ground Truth: positive, Sentence: Fabulous service, fantastic food, and a chilled out atmosphere and environment."
    },
    {
        "Rule": "direct_word_positive",
        "Precision": 0.7222222222222222,
        "Recall": 0.7222222222222222,
        "Successful Cases": "- Aspect: cheese, Sentiment: positive, Ground Truth: positive, Sentence: If you love wine and cheese and delicious french fare, you'll love Artisanal!",
        "Failure Cases": "- Aspect: room, Sentiment: positive, Ground Truth: neutral, Sentence: Looking around, I saw a room full of New Yorkers enjoying a real meal in a real restaurant, not a clubhouse of the fabulous trying to be seen."
    },
    {
        "Rule": "adverb_neutral",
        "Precision": 1.0,
        "Recall": 0.30434782608695654,
        "Successful Cases": "- Aspect: food, Sentiment: neutral, Ground Truth: neutral, Sentence: Decor is nice and minimalist, food simple yet very well presented and cooked, and the wine list matches the food very well.",
        "Failure Cases": "- Aspect: waiting, Sentiment: neutral, Ground Truth: negative, Sentence: Overall A oh ya even though there is waiting it is deff worth it"
    },
    {
        "Rule": "adverb_positive",
        "Precision": 0.9565217391304348,
        "Recall": 0.9565217391304348,
        "Successful Cases": "- Aspect: served, Sentiment: positive, Ground Truth: positive, Sentence: The food always tastes fresh and served promptly.",
        "Failure Cases": "- Aspect: served, Sentiment: positive, Ground Truth: negative, Sentence: This place would be so much better served by being run by a group that actually understands customer service."
    },
    {
        "Rule": "negation",
        "Precision": 0.2608695652173913,
        "Recall": 0.2608695652173913,
        "Successful Cases": "- Aspect: food, Sentiment: neutral, Ground Truth: neutral, Sentence: not the food ,not the ambiance , not the service, I agree with the previous reviews you wait and wait , the wait staff are very rude and when you get in they are looking to get you right out .",
        "Failure Cases": "- Aspect: place, Sentiment: positive, Ground Truth: conflict, Sentence: Not a large place, but it's cute and cozy."
    },
    {
        "Rule": "direct_word_negative",
        "Precision": 0.35714285714285715,
        "Recall": 0.35714285714285715,
        "Successful Cases": "- Aspect: waiter, Sentiment: negative, Ground Truth: negative, Sentence: One would think we'd get an apology or complimentary drinks - instead, we got a snobby waiter wouldn't even take our order for 15 minutes and gave us lip when we asked him to do so.",
        "Failure Cases": "- Aspect: spot, Sentiment: negative, Ground Truth: positive, Sentence: Metrazur has a beautiful spot overlooking the main terminal."
    },
    {
        "Rule": "adverb_negative",
        "Precision": 1.0,
        "Recall": 1.0,
        "Successful Cases": "- Aspect: cooked, Sentiment: negative, Ground Truth: negative, Sentence: The rice was poor quality and was cooked so badly it was hard.",
        "Failure Cases": "No failure cases."
    }
]

df = pd.DataFrame(data)

fig = go.Figure(data=[go.Table(header=dict(values=df.columns),
                               cells=dict(values=[df[col] for col in df.columns]))
                      ])

fig.update_layout(title="Data Table")
fig.show()
