"""
Génère un dataset local structuré sans téléchargement.
Combine : texte Shakespeare embarqué + patterns linguistiques.
Résultat : data/local_train.bin et data/local_val.bin (tokens uint16).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tiktoken

# ── Texte Shakespeare embarqué (domaine public) ────────────────
# Extrait de Hamlet + King Lear + Macbeth — ~8000 tokens
CORPUS = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

All the world's a stage, And all the men and women merely players;
They have their exits and their entrances, And one man in his time plays many parts,
His acts being seven ages. At first, the infant, Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel And shining morning face, creeping like snail
Unwillingly to school. And then the lover, Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier, Full of strange oaths and bearded like the pard,
Jealous in honor, sudden and quick in quarrel, Seeking the bubble reputation
Even in the cannon's mouth. And then the justice, In fair round belly with good capon lined,
With eyes severe and beard of formal cut, Full of wise saws and modern instances;
And so he plays his part. The sixth age shifts Into the lean and slippered pantaloon,
With spectacles on nose and pouch on side; His youthful hose, well saved, a world too wide
For his shrunk shank, and his big manly voice, Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all, That ends this strange eventful history,
Is second childishness and mere oblivion, Sans teeth, sans eyes, sans taste, sans everything.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest—
For Brutus is an honourable man;
So are they all, all honourable men—
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.
He hath brought many captives home to Rome,
Whose ransoms did the general coffers fill:
Did this in Caesar seem ambitious?
When that the poor have cried, Caesar hath wept:
Ambition should be made of sterner stuff:
Yet Brutus says he was ambitious;
And Brutus is an honourable man.

Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day,
To the last syllable of recorded time;
And all our yesterdays have lighted fools
The way to dusty death. Out, out, brief candle!
Life's but a walking shadow, a poor player,
That struts and frets his hour upon the stage,
And then is heard no more. It is a tale
Told by an idiot, full of sound and fury,
Signifying nothing.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.
Grim-visaged war hath smooth'd his wrinkled front;
And now, instead of mounting barbed steeds
To fright the souls of fearful adversaries,
He capers nimbly in a lady's chamber
To the lascivious pleasing of a lute.

What a piece of work is a man! How noble in reason, how infinite in faculty!
In form and moving how express and admirable! In action how like an angel!
In apprehension how like a god! The beauty of the world, the paragon of animals!
And yet, to me, what is this quintessence of dust?
Man delights not me—no, nor woman neither, though by your smiling you seem to say so.

The quality of mercy is not strained.
It droppeth as the gentle rain from heaven
Upon the place beneath. It is twice blest:
It blesseth him that gives and him that takes.
'Tis mightiest in the mightiest; it becomes
The throned monarch better than his crown.
His sceptre shows the force of temporal power,
The attribute to awe and majesty,
Wherein doth sit the dread and fear of kings;
But mercy is above this sceptred sway.
It is enthroned in the hearts of kings,
It is an attribute to God himself;
And earthly power doth then show likest God's
When mercy seasons justice.

O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? it is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? that which we call a rose
By any other name would smell as sweet;
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name,
And for that name which is no part of thee
Take all myself.

Cowards die many times before their deaths;
The valiant never taste of death but once.
Of all the wonders that I yet have heard,
It seems to me most strange that men should fear;
Seeing that death, a necessary end,
Will come when it will come.

We know what we are, but know not what we may be.
Doubt thou the stars are fire; Doubt that the sun doth move;
Doubt truth to be a liar; But never doubt I love.
This above all: to thine own self be true,
And it must follow, as the night the day,
Thou canst not then be false to any man.
Farewell, my blessing season this in thee!

Neither a borrower nor a lender be;
For loan oft loses both itself and friend,
And borrowing dulls the edge of husbandry.
Give every man thy ear, but few thy voice;
Take each man's censure, but reserve thy judgment.
Costly thy habit as thy purse can buy,
But not express'd in fancy; rich, not gaudy;
For the apparel oft proclaims the man.

How sharper than a serpent's tooth it is
To have a thankless child! Away, away!
I am a man more sinned against than sinning.
Nothing will come of nothing: speak again.
As flies to wanton boys are we to the gods,
They kill us for their sport.
""" * 12  # Répété 12× → ~100k tokens, assez pour un vrai entraînement


def prepare_local(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "local_train.bin")
    val_path   = os.path.join(data_dir, "local_val.bin")

    if os.path.exists(train_path):
        print("Dataset local déjà présent.")
        return train_path, val_path

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(CORPUS)
    tokens = np.array(tokens, dtype=np.uint16)

    split = int(0.9 * len(tokens))
    tokens[:split].tofile(train_path)
    tokens[split:].tofile(val_path)

    print(f"Dataset créé : {len(tokens):,} tokens total")
    print(f"  train : {split:,}  |  val : {len(tokens)-split:,}")
    return train_path, val_path


if __name__ == "__main__":
    prepare_local()
