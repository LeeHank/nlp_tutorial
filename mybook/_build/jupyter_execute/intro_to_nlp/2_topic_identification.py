#!/usr/bin/env python
# coding: utf-8

# # Simple topic identification

# In[27]:


import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import re

import pandas as pd
import numpy as np


# ## bag-of-words

# * 這邊要來練習，斷詞後，製作詞頻。
# * 以下以 `article` 這個字串為例，從斷詞到製作詞頻走一次：

# In[1]:


article = '\'\'\'Debugging\'\'\' is the process of finding and resolving of defects that prevent correct operation of computer software or a system.  \n\nNumerous books have been written about debugging (see below: #Further reading|Further reading), as it involves numerous aspects, including interactive debugging, control flow, integration testing, Logfile|log files, monitoring (Application monitoring|application, System Monitoring|system), memory dumps, Profiling (computer programming)|profiling, Statistical Process Control, and special design tactics to improve detection while simplifying changes.\n\nOrigin\nA computer log entry from the Mark&nbsp;II, with a moth taped to the page\n\nThe terms "bug" and "debugging" are popularly attributed to Admiral Grace Hopper in the 1940s.[http://foldoc.org/Grace+Hopper Grace Hopper]  from FOLDOC While she was working on a Harvard Mark II|Mark II Computer at Harvard University, her associates discovered a moth stuck in a relay and thereby impeding operation, whereupon she remarked that they were "debugging" the system. However the term "bug" in the meaning of technical error dates back at least to 1878 and Thomas Edison (see software bug for a full discussion), and "debugging" seems to have been used as a term in aeronautics before entering the world of computers. Indeed, in an interview Grace Hopper remarked that she was not coining the term{{Citation needed|date=July 2015}}. The moth fit the already existing terminology, so it was saved.  A letter from J. Robert Oppenheimer (director of the WWII atomic bomb "Manhattan" project at Los Alamos, NM) used the term in a letter to Dr. Ernest Lawrence at UC Berkeley, dated October 27, 1944,http://bancroft.berkeley.edu/Exhibits/physics/images/bigscience25.jpg regarding the recruitment of additional technical staff.\n\nThe Oxford English Dictionary entry for "debug" quotes the term "debugging" used in reference to airplane engine testing in a 1945 article in the Journal of the Royal Aeronautical Society. An article in "Airforce" (June 1945 p.&nbsp;50) also refers to debugging, this time of aircraft cameras.  Hopper\'s computer bug|bug was found on September 9, 1947. The term was not adopted by computer programmers until the early 1950s.\nThe seminal article by GillS. Gill, [http://www.jstor.org/stable/98663 The Diagnosis of Mistakes in Programmes on the EDSAC], Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, Vol. 206, No. 1087 (May 22, 1951), pp. 538-554 in 1951 is the earliest in-depth discussion of programming errors, but it does not use the term "bug" or "debugging".\nIn the Association for Computing Machinery|ACM\'s digital library, the term "debugging" is first used in three papers from 1952 ACM National Meetings.Robert V. D. Campbell, [http://portal.acm.org/citation.cfm?id=609784.609786 Evolution of automatic computation], Proceedings of the 1952 ACM national meeting (Pittsburgh), p 29-32, 1952.Alex Orden, [http://portal.acm.org/citation.cfm?id=609784.609793 Solution of systems of linear inequalities on a digital computer], Proceedings of the 1952 ACM national meeting (Pittsburgh), p. 91-95, 1952.Howard B. Demuth, John B. Jackson, Edmund Klein, N. Metropolis, Walter Orvedahl, James H. Richardson, [http://portal.acm.org/citation.cfm?id=800259.808982 MANIAC], Proceedings of the 1952 ACM national meeting (Toronto), p. 13-16 Two of the three use the term in quotation marks.\nBy 1963 "debugging" was a common enough term to be mentioned in passing without explanation on page 1 of the Compatible Time-Sharing System|CTSS manual.[http://www.bitsavers.org/pdf/mit/ctss/CTSS_ProgrammersGuide.pdf The Compatible Time-Sharing System], M.I.T. Press, 1963\n\nKidwell\'s article \'\'Stalking the Elusive Computer Bug\'\'Peggy Aldrich Kidwell, [http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?tp=&arnumber=728224&isnumber=15706 Stalking the Elusive Computer Bug], IEEE Annals of the History of Computing, 1998. discusses the etymology of "bug" and "debug" in greater detail.\n\nScope\nAs software and electronic systems have become generally more complex, the various common debugging techniques have expanded with more methods to detect anomalies, assess impact, and schedule software patches or full updates to a system. The words "anomaly" and "discrepancy" can be used, as being more neutral terms, to avoid the words "error" and "defect" or "bug" where there might be an implication that all so-called \'\'errors\'\', \'\'defects\'\' or \'\'bugs\'\' must be fixed (at all costs). Instead, an impact assessment can be made to determine if changes to remove an \'\'anomaly\'\' (or \'\'discrepancy\'\') would be cost-effective for the system, or perhaps a scheduled new release might render the change(s) unnecessary. Not all issues are life-critical or mission-critical in a system. Also, it is important to avoid the situation where a change might be more upsetting to users, long-term, than living with the known problem(s) (where the "cure would be worse than the disease"). Basing decisions of the acceptability of some anomalies can avoid a culture of a "zero-defects" mandate, where people might be tempted to deny the existence of problems so that the result would appear as zero \'\'defects\'\'. Considering the collateral issues, such as the cost-versus-benefit impact assessment, then broader debugging techniques will expand to determine the frequency of anomalies (how often the same "bugs" occur) to help assess their impact to the overall system.\n\nTools\nDebugging on video game consoles is usually done with special hardware such as this Xbox (console)|Xbox debug unit intended for developers.\n\nDebugging ranges in complexity from fixing simple errors to performing lengthy and tiresome tasks of data collection, analysis, and scheduling updates.  The debugging skill of the programmer can be a major factor in the ability to debug a problem, but the difficulty of software debugging varies greatly with the complexity of the system, and also depends, to some extent, on the programming language(s) used and the available tools, such as \'\'debuggers\'\'. Debuggers are software tools which enable the programmer to monitor the execution (computers)|execution of a program, stop it, restart it, set breakpoints, and change values in memory. The term \'\'debugger\'\' can also refer to the person who is doing the debugging.\n\nGenerally, high-level programming languages, such as Java (programming language)|Java, make debugging easier, because they have features such as exception handling that make real sources of erratic behaviour easier to spot. In programming languages such as C (programming language)|C or assembly language|assembly, bugs may cause silent problems such as memory corruption, and it is often difficult to see where the initial problem happened. In those cases, memory debugging|memory debugger tools may be needed.\n\nIn certain situations, general purpose software tools that are language specific in nature can be very useful.  These take the form of \'\'List of tools for static code analysis|static code analysis tools\'\'.  These tools look for a very specific set of known problems, some common and some rare, within the source code.  All such issues detected by these tools would rarely be picked up by a compiler or interpreter, thus they are not syntax checkers, but more semantic checkers.  Some tools claim to be able to detect 300+ unique problems. Both commercial and free tools exist in various languages.  These tools can be extremely useful when checking very large source trees, where it is impractical to do code walkthroughs.  A typical example of a problem detected would be a variable dereference that occurs \'\'before\'\' the variable is assigned a value.  Another example would be to perform strong type checking when the language does not require such.  Thus, they are better at locating likely errors, versus actual errors.  As a result, these tools have a reputation of false positives.  The old Unix \'\'Lint programming tool|lint\'\' program is an early example.\n\nFor debugging electronic hardware (e.g., computer hardware) as well as low-level software (e.g., BIOSes, device drivers) and firmware, instruments such as oscilloscopes, logic analyzers or in-circuit emulator|in-circuit emulators (ICEs) are often used, alone or in combination.  An ICE may perform many of the typical software debugger\'s tasks on low-level software and firmware.\n\nDebugging process \nNormally the first step in debugging is to attempt to reproduce the problem. This can be a non-trivial task, for example as with Parallel computing|parallel processes or some unusual software bugs. Also, specific user environment and usage history can make it difficult to reproduce the problem.\n\nAfter the bug is reproduced, the input of the program may need to be simplified to make it easier to debug. For example, a bug in a compiler can make it Crash (computing)|crash when parsing some large source file. However, after simplification of the test case, only few lines from the original source file can be sufficient to reproduce the same crash. Such simplification can be made manually, using a Divide and conquer algorithm|divide-and-conquer approach. The programmer will try to remove some parts of original test case and check if the problem still exists. When debugging the problem in a Graphical user interface|GUI, the programmer can try to skip some user interaction from the original problem description and check if remaining actions are sufficient for bugs to appear.\n\nAfter the test case is sufficiently simplified, a programmer can use a debugger tool to examine program states (values of variables, plus the call stack) and track down the origin of the problem(s). Alternatively, Tracing (software)|tracing can be used. In simple cases, tracing is just a few print statements, which output the values of variables at certain points of program execution.{{citation needed|date=February 2016}}\n\n Techniques \n \'\'Interactive debugging\'\'\n \'\'{{visible anchor|Print debugging}}\'\' (or tracing) is the act of watching (live or recorded) trace statements, or print statements, that indicate the flow of execution of a process. This is sometimes called \'\'{{visible anchor|printf debugging}}\'\', due to the use of the printf function in C. This kind of debugging was turned on by the command TRON in the original versions of the novice-oriented BASIC programming language. TRON stood for, "Trace On." TRON caused the line numbers of each BASIC command line to print as the program ran.\n \'\'Remote debugging\'\' is the process of debugging a program running on a system different from the debugger. To start remote debugging, a debugger connects to a remote system over a network. The debugger can then control the execution of the program on the remote system and retrieve information about its state.\n \'\'Post-mortem debugging\'\' is debugging of the program after it has already Crash (computing)|crashed. Related techniques often include various tracing techniques (for example,[http://www.drdobbs.com/tools/185300443 Postmortem Debugging, Stephen Wormuller, Dr. Dobbs Journal, 2006]) and/or analysis of memory dump (or core dump) of the crashed process. The dump of the process could be obtained automatically by the system (for example, when process has terminated due to an unhandled exception), or by a programmer-inserted instruction, or manually by the interactive user.\n \'\'"Wolf fence" algorithm:\'\' Edward Gauss described this simple but very useful and now famous algorithm in a 1982 article for communications of the ACM as follows: "There\'s one wolf in Alaska; how do you find it? First build a fence down the middle of the state, wait for the wolf to howl, determine which side of the fence it is on. Repeat process on that side only, until you get to the point where you can see the wolf."<ref name="communications of the ACM">{{cite journal | title="Pracniques: The "Wolf Fence" Algorithm for Debugging", | author=E. J. Gauss | year=1982}} This is implemented e.g. in the Git (software)|Git version control system as the command \'\'git bisect\'\', which uses the above algorithm to determine which Commit (data management)|commit introduced a particular bug.\n \'\'Delta Debugging\'\'{{snd}} a technique of automating test case simplification.Andreas Zeller: <cite>Why Programs Fail: A Guide to Systematic Debugging</cite>, Morgan Kaufmann, 2005. ISBN 1-55860-866-4{{rp|p.123}}<!-- for redirect from \'Saff Squeeze\' -->\n \'\'Saff Squeeze\'\'{{snd}} a technique of isolating failure within the test using progressive inlining of parts of the failing test.[http://www.threeriversinstitute.org/HitEmHighHitEmLow.html Kent Beck, Hit \'em High, Hit \'em Low: Regression Testing and the Saff Squeeze]\n\nDebugging for embedded systems\nIn contrast to the general purpose computer software design environment, a primary characteristic of embedded environments is the sheer number of different platforms available to the developers (CPU architectures, vendors, operating systems and their variants). Embedded systems are, by definition, not general-purpose designs: they are typically developed for a single task (or small range of tasks), and the platform is chosen specifically to optimize that application. Not only does this fact make life tough for embedded system developers, it also makes debugging and testing of these systems harder as well, since different debugging tools are needed in different platforms.\n\nto identify and fix bugs in the system (e.g. logical or synchronization problems in the code, or a design error in the hardware);\nto collect information about the operating states of the system that may then be used to analyze the system: to find ways to boost its performance or to optimize other important characteristics (e.g. energy consumption, reliability, real-time response etc.).\n\nAnti-debugging\nAnti-debugging is "the implementation of one or more techniques within computer code that hinders attempts at reverse engineering or debugging a target process".<ref name="veracode-antidebugging">{{cite web |url=http://www.veracode.com/blog/2008/12/anti-debugging-series-part-i/ |title=Anti-Debugging Series - Part I |last=Shields |first=Tyler |date=2008-12-02 |work=Veracode |accessdate=2009-03-17}} It is actively used by recognized publishers in copy protection|copy-protection schemas, but is also used by malware to complicate its detection and elimination.<ref name="soft-prot">[http://people.seas.harvard.edu/~mgagnon/software_protection_through_anti_debugging.pdf Software Protection through Anti-Debugging Michael N Gagnon, Stephen Taylor, Anup Ghosh] Techniques used in anti-debugging include:\nAPI-based: check for the existence of a debugger using system information\nException-based: check to see if exceptions are interfered with\nProcess and thread blocks: check whether process and thread blocks have been manipulated\nModified code: check for code modifications made by a debugger handling software breakpoints\nHardware- and register-based: check for hardware breakpoints and CPU registers\nTiming and latency: check the time taken for the execution of instructions\nDetecting and penalizing debugger<ref name="soft-prot" /><!-- reference does not exist -->\n\nAn early example of anti-debugging existed in early versions of Microsoft Word which, if a debugger was detected, produced a message that said: "The tree of evil bears bitter fruit. Now trashing program disk.", after which it caused the floppy disk drive to emit alarming noises with the intent of scaring the user away from attempting it again.<ref name="SecurityEngineeringRA">{{cite book | url=http://www.cl.cam.ac.uk/~rja14/book.html | author=Ross J. Anderson | title=Security Engineering | isbn = 0-471-38922-6 | page=684 }}<ref name="toastytech">{{cite web | url=http://toastytech.com/guis/word1153.html | title=Microsoft Word for DOS 1.15}}\n'


# In[6]:


# 斷詞
tokens = word_tokenize(article)

# 將結果轉小寫
lower_tokens = [t.lower() for t in tokens]

lower_tokens[0:10]


# * 建立詞頻，會用到內建的 `collections.Counter()`，作法如下：

# In[19]:


from collections import Counter
bow_simple = Counter(lower_tokens)


# * 做完的物件，是一個 counter 物件 (可以想成 dictionary 就好)，裡面就是 key-value pair (key對應到該詞，value就是詞頻)  
# * 我們可以用他的 `.most_comon()` method，來找出最多的詞頻

# In[24]:


bow_simple.most_common(10)


# * 可以看到，最高頻的是 `,`，出現 151 次。再來是 `the`，出現 150 次  
# * 如果你很不喜歡這種格式，可以這樣轉成 pandas

# In[29]:


pd.DataFrame(bow_simple.items(), columns = ["key", "value"])


# ## text preprocessing

# * 要做出比較好的詞頻表，通常都會經過以下的前處理過程：  
#   * 把字體全改小寫 (用 str 的 method: `.lower()`). 
#   * 只取出文字(不要標點). 
#   * 將動詞三態, 名詞單複數全還原 (lemmatizer). 
#   * 排除掉 stop words. 
# * 以下，拿剛剛的 `article`，再做一次，但加入上述多個 preprocessing：

# In[35]:


nltk.download('wordnet')
nltk.download('omw-1.4')


# In[36]:


# 斷詞
tokens = word_tokenize(article)

# 將結果轉小寫
lower_tokens = [t.lower() for t in tokens]

# 只留文字
alpha_only = [t for t in lower_tokens if t.isalpha()]

# 移除 stop words
english_stops = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', '']
no_stops = [t for t in alpha_only if t not in english_stops]

# 將動詞三態, 名詞單複數全還原 (lemmatizer)
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# 建立 bag-of-words
from collections import Counter
bow = Counter(lemmatized)

# print 出最高頻的 10 個字
print(bow.most_common(10))


# ## intro to `gensim`

# * `gensim` 是一個 popular open-source NLP library. 
# * 他可以做到：  
#   * building document or word vectors. 
#   * performing topic identification and document comparison. 

# ### 範例資料

# * 先來個範例資料：在 `data/Wikipedia articles` 裡面，有 12 個檔案 (12篇文章)
# * 我把每個文章打開後，斷詞與前處理，再存到 list 裡：

# In[70]:


import os
wiki_files = os.listdir('data/Wikipedia articles')

def text_preprocessing(doc):
    # 斷詞
    tokens = word_tokenize(doc)

    # 將結果轉小寫
    lower_tokens = [t.lower() for t in tokens]

    # 只留文字
    alpha_only = [t for t in lower_tokens if t.isalpha()]

    # 移除 stop words
    english_stops = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', '']
    no_stops = [t for t in alpha_only if t not in english_stops]
    
    return no_stops

articles = []
for i in wiki_files:    
    with open("data/Wikipedia articles/"+i, "r+", encoding = "UTF-8") as f:
        doc = f.read()
        articles.append(text_preprocessing(doc))


# * 來看一下第 12 篇文章，斷完詞後，前 10 個字是哪些：

# In[73]:


articles[11][0:10]


# ### id vs token 對應表

# * 有了 `articles` 這種資料格式後 (兩層的 list)，就可以餵到 gensim 的 `Dictionary` 裡面，做出 id vs token 的對應表：

# In[75]:


from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(articles)


# * 這個字典，是一個 iterator 物件，直接打開看不到東西：

# In[97]:


dictionary


# * 我們用迴圈解開來看的話，前10筆長這樣：

# In[96]:


for key, value in dictionary.items():
    if key < 10:
        print(str(key) + " -- " + str(value))


# * 可以得知，這個 dictionary 是 {"id": "token"} 這種組合。  
# * 所以我如果想輸入 id ，查他是哪個字，我可以這樣做：

# In[100]:


dictionary.get(1)


# * 反過來，我如果知道字，想知道他的 id 呢？ 那要用 `dictionary.token2id` 這個 attribute

# In[101]:


dictionary.token2id


# * 所以，我想查 "computer" 這個字的 id ，我可以這樣做：

# In[102]:


dictionary.token2id.get("computer")


# * 如果，我要查 "computer" 的 id，我可以這樣查：

# In[84]:


computer_id = dictionary.token2id.get("computer")
computer_id


# In[98]:


dictionary.get(computer_id)


# ### bag of word

# * 有了剛剛的 id vs token 對應表後 (dictionary)，可以使用 `.doc2bow()` 這個 method，去看一篇文章中，各個 id 出現幾次. 

# In[118]:


corpus = [dictionary.doc2bow(article) for article in articles]


# * 例如，我看一下剛剛的第 5 篇文章，他的 bag of word:

# In[124]:


bow_4 = corpus[4]
bow_4[0:5]


# * 可以看到，我列出前五筆，他分別告訴我， id 是 1 的字，出現 1 次; id 是 13 的字，出現 1 次 ...  
# * 他只會列出有出現的字，所以 `bow_4` 只列出 706 個字的詞頻。但原本的字典有 6211 個字

# In[125]:


print(len(bow_4))
print(len(dictionary.keys()))


# * 那我如果想看，這篇文章最高頻的 5 個字，我可以這樣做：

# In[128]:


bow_4_sort = sorted(bow_4, key=lambda w: w[1], reverse=True)
for word_id, word_count in bow_4_sort[:5]:
    print(dictionary.get(word_id), word_count)


# * 如果要做出所有 article 的 bow (就全部 article 一起看的意思)

# In[132]:


from collections import defaultdict
import itertools

total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count


# * 那就可以得到以下 dictionary

# In[133]:


total_word_count


# * 依照詞頻做排序：

# In[135]:


sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True) 


# * 取最高頻前五名：

# In[136]:


for word_id, word_count in sorted_word_count[:5]:
    print(dictionary.get(word_id), word_count)


# * 跟之前一樣，如果你對他的資料格式很不爽 (對 dictionary 或 list of tuple 的格式每送，比較喜歡dataframe) ，那可以自己做轉換：

# In[143]:


bow_4_words = [dictionary.get(i[0]) for i in bow_4]
bow_4_df = pd.concat(
    [pd.DataFrame(bow_4_words, columns = ["word"]),
    pd.DataFrame(bow_4, columns = ["id", "freq"])],
    axis = 1
)
bow_4_df.sort_values("freq", ascending = False)


# * 所有文章一起看詞頻df也可以：

# In[142]:


total_words = [dictionary.get(i[0]) for i in total_word_count.items()]
total_df = pd.concat(
    [pd.DataFrame(total_words, columns = ["word"]),
    pd.DataFrame(total_word_count.items(), columns = ["id", "freq"])],
    axis = 1
)
total_df.sort_values("freq", ascending=False)


# ### tf-idf

# * tf-idf 的算法是：  
#   * $tf_{ij} = \frac{n_{ij}}{\sum_k n_{kj}}$: 第 i 個詞，在第j個文件中的tf，等於第i個詞在第j個文件的詞頻($n_{ij}$，除上第i個詞在所有文件的總詞頻。  
#   * $idf_{i} = log(\frac{N}{1 + df_{i}})$: 第 i 個詞的 inverse document frequency，是用 總文件數 除以 (1 + 共多少文件出現過第i個詞)，再取 log. 
# * 所以，第 j 個文件中的第 i 個詞，他的 tf-idf = $tf_{ij} \times idf_{i}$
# * 那我們在 traning 的時候，就是先把 tf 中的 $\sum_k n_{kj}$ 的算好，把 $idf_{i}$ 算好，那只要丟一份新文件給我時，告訴我 $n_{ij}$，就可以幫你算出 tf-idf 了
# * 舉例來說，我們先用剛剛的 `corpus` 來 train 一個 `tfidf` model

# In[154]:


from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus)


# * 那現在有一個新的文件，他的詞頻長這樣：

# In[155]:


my_doc = [(1, 5), (13, 3), (15, 1), (18, 1)]


# * 他的 tf-idf 就會是：

# In[156]:


tfidf[my_doc]


# * 同理，我如果要拿之前第5篇的文章 (bow_4) 當例子，他的 tf-idf 就會是

# In[160]:


tfidf_weights = tfidf[bow_4]
tfidf_weights[:5] # 取前五筆來呈現就好


# * 那我如果想知道這篇文章的關鍵字，我就可以看 tf-idf 最大的五個詞是哪 5 個 (就不會只看詞頻最大的 5 個了)

# In[161]:


sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)

# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]:
    print(dictionary.get(term_id), weight)

