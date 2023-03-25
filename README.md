# nanoGPT

## Project Intention
In diesem ML Projekt baue ich Andrey Karpathy's nanoGPT nach:

* um Erfahrungen mit dem Trainieren solcher Modelle zu sammeln
* um ein besseres Verständnis von LLMs zu bekommen
* die Version vielleicht irgendwann mal zu erweitern
* die Version irgendwann mal für andere Zwecke verwenden zu können 

## Dataset

Mit diesem GPT wird versucht Shakespeare-like Language zu produzieren, also Sprache, die der von Shakespeare ähnlich ist. Es ist aber auch möglich jede beliebige ähnliche Daten zu reproduzieren.

Das Dataset - Tiny Shakesspeare - lässt sich mit `wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` runterladen. Für ein Preview siehe `input.txt`.

## Model Achitecture

Dieses Modell arbeitet auf einem 1-Character Token Level. Es macht also immer Vorhersagen über den nächsten kommenden Buchstaben. *Ausführung folgt...*

## Model Specs

Mit der aktuellsten Skalierung hat der GPT eine Context Size von 256 Characters, und 384 Feature Channels. Er ist ein 6-Layer Transformer mit 6 Heads pro Layer.

```
# hyperparameters
batch_size = 64 # how many independent sequences are processed in parallel
block_size = 256 # the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # number of embedding dimensions
n_head = 6
n_layer = 6
dropout = 0.2
```

## Output

Mit der aktuellen Skalierung kommt man nach dem Training auf 1.49 Validation Loss:

```
step 0: train loss 4.2849, val loss 4.2823
step 500: train loss 1.9976, val loss 2.0895
step 1000: train loss 1.5957, val loss 1.7595
step 1500: train loss 1.4335, val loss 1.6423
step 2000: train loss 1.3384, val loss 1.5673
step 2500: train loss 1.2771, val loss 1.5301
step 3000: train loss 1.2230, val loss 1.5077
step 3500: train loss 1.1835, val loss 1.4857
step 4000: train loss 1.1418, val loss 1.4787
step 4500: train loss 1.1046, val loss 1.4849
step 4999: train loss 1.0722, val loss 1.4908
```

Der Output ähnelt den Trainingsdaten schon sehr, lässt sich aber natürlich verbessern, wenn man das Modell hochskaliert. 

Example Generation:

```
ESCALE:
Ay, your honour show shall be so fall'n.

ESCALUS:
Why, inducation Paris hath all that
Pity For ivytue sufficiency, were your mother
Must lost in disturbon.'

ESCURRAL:
So, o I am; the composing is
Not for them that your posper in their mother:
What you would lie you to, too far with your grace?

Nurse:
I am battled before it, as device, mighty insture of
him, in you mass, be we abal-made take,
disperfulness, condemn the death of dangments?
O errassing, good conspirator, welcome,
To see't no match, in honour.
She can raged sweet Montagues be a courteous counse!
Come, fellow, troot: wretch, come! worst thou flat'st mine!
And these nobles talk apace well appeacheth, all
Proceptre to know the Antiament haths.
But dore the groanies shall not slips.
There now more their voices; nor no longinger.
Even live would not shower to run them in
them is sobstools to be to bed, sir, is that they
histood.
```

Für die ersten 10000 Characters des produzierten Outputs siehe `more.txt`.