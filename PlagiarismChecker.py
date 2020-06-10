import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


# Original file
f = open(os.path.abspath(os.path.dirname(os.path.abspath(__file__)))+'/Data/Original.txt',"r")
original_f = f.read().replace("\n"," ")
f.close()

# Suspicious file
f2 = open(os.path.abspath(os.path.dirname(os.path.abspath(__file__)))+'/Data/CC8.txt',"r")
plag_f=f2.read().replace("\n"," ")
f2.close()
#Tokenize both original and suspicious file
tokens_orig = word_tokenize(original_f)
tokens_plag = word_tokenize(plag_f)

# convert tokens to lowerCase
tokens_orig = [token.lower() for token in tokens_orig]
tokens_plag = [token.lower() for token in tokens_plag]

#remove stop words and punctuation
stop_words=set(stopwords.words('english'))
punctuations=['"','.','(',')',',','?',';',':',"''",'``']

preproc_orig = [w for w in tokens_orig if not w in stop_words and not w in punctuations]
preproc_plag = [w for w in tokens_plag if not w in stop_words and not w in punctuations]

#lemmatise all the words in tokens
lemmatizer = WordNetLemmatizer()

orig_final = []
plag_final = []
for w in preproc_orig:
    w = lemmatizer.lemmatize(w)
    orig_final.append(w)

for w in preproc_plag:
    w = lemmatizer.lemmatize(w)
    plag_final.append(w)

# Trigram for original text
trigrams_orig = []
for i in range(len(orig_final) - 2):
    t = (orig_final[i], orig_final[i + 1], orig_final[i + 2])
    trigrams_orig.append(t)

# s will contain the number of intersected trigrams from original and suspicious text
s = 0
# Trigram for suspicious text
trigrams_plag = []
for i in range(len(plag_final) - 2):
    t = (plag_final[i], plag_final[i + 1], plag_final[i + 2])
    trigrams_plag.append(t)
    if t in trigrams_orig:
        s += 1

# jaccard coefficient = (S(original)((intersection))S(plagiarised)) / (S(original) U S(plagiarised))
J = s / (len(trigrams_orig) + len(trigrams_plag))
print("Jaccard coefficient:: ", J)

# containment score = (S(original)((intersection))S(plagiarised)) / S(plagiarised)
C = s / len(trigrams_plag)
print("Containment score:: ", C)


# longest common subsequence
def lcs(l1, l2):
    s1 = word_tokenize(l1)
    s2 = word_tokenize(l2)
    dp_lcs = [[None] * (len(s1) + 1) for i in range(len(s2) + 1)]

    for i in range(len(s2) + 1):
        for j in range(len(s1) + 1):
            if i == 0 or j == 0:
                dp_lcs[i][j] = 0
            elif s2[i - 1] == s1[j - 1]:
                dp_lcs[i][j] = dp_lcs[i - 1][j - 1] + 1
            else:
                dp_lcs[i][j] = max(dp_lcs[i - 1][j], dp_lcs[i][j - 1])
    return dp_lcs[len(s2)][len(s1)]


sent_o = sent_tokenize(original_f)
sent_p = sent_tokenize(plag_f)

#remove stopwords from sentences
sent_p_clean = []
sent_o_clean = []
for line in sent_o:
    tokens = word_tokenize(line)
    item = [w for w in tokens if not w in stop_words]
    line_f = ' '.join(item)
    sent_o_clean.append(line_f)

for line in sent_p:
    tokens = word_tokenize(line)
    item = [w for w in tokens if not w in stop_words]
    line_f = ' '.join(item)
    sent_p_clean.append(line_f)

# maximum length of LCS for a sentence in suspicious text
max_l = 0
sum_lcs = 0

for i in sent_p:
    for j in sent_o:
        l = lcs(i, j)
        max_l = max(max_l, l)
    sum_lcs += max_l
    max_l = 0

score = sum_lcs / len(tokens_plag)
print("LCS Score:: ", score)

#Give weightage to each score:: 1.5 to Jaccard, 3.5 to Containment and 5 to LCS
J_w = 1.5 * J
C_w = 3.5 * C
LCS_w = 5* score
final_score = (J_w + C_w + LCS_w) / 10
#print(final_score)

if final_score >= 0.7:
    print("Near copy")
elif (final_score >= 0.4):
    if (final_score <0.7):
        print("Lightly revised")
elif (final_score >= 0.2):
    if (final_score < 0.4):
        print("Heavily revised")
else:
    print("Not plagiarised")