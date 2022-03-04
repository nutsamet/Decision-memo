# Decision-memo

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Set up our Cohen's d function
def cohens_d (x,y):
    n1 = x.shape[0]
    n2 = y.shape[0]
    sd = (((n1 - 1) * x.var() + (n2 - 1) * y.var())/(n1 + n2 - 2))**0.5
    return ((x.mean()-y.mean())/sd)

# Define more concise headers for the data
headers = ["timestamp","level","country","is_online",
           "first_survey","agree","primary_mode","preferred_mode",
           "why_mode","platforms_used","methods_used",
           "techniques_outside","remote_enjoy","remote_motivated",
           "remote_satisfied","remote_engaging","remote_distracted",
           "remote_questions","remote_changes","prior_enjoy",
           "prior_motivated","prior_satisfied","prior_engaging",
           "prior_distracted","prior_questions","prior_changes",
           "preference","why_preference"]

# Read in the raw data
data = pd.read_csv("RLData.csv", skiprows=[0], names=headers, na_values="?")
# Define some other lists that will be useful

# List of just the Likert questions about remote instructions
remote_survey_q = ["remote_enjoy","remote_motivated",
                   "remote_satisfied","remote_engaging",
                   "remote_distracted","remote_questions"]
change_survey_q = ["change_enjoy","change_motivated",
                   "change_satisfied","change_engaging",
                   "change_distracted","change_questions"]
remote_q_long = ["I enjoy having courses online",
                 "I feel motivated to learn",
                 "I am satisfied with the instruction\nof my online courses",
                 "My courses are engaging",
                 "I am often distracted when doing\ncourse work / attending classes",
                 "I often ask questions, comment, join discussions"]
change_q_long = ["I enjoy",
                 "I feel motivated to learn",
                 "I am satisfied with the instruction",
                 "My courses are engaging",
                 "I am often distracted when doing\ncourse work / attending classes",
                 "I often ask questions, comment, join discussions"]

# Drop the rows that we're not going to be using
data_2 = data.drop(["timestamp","agree","why_mode",
                        "remote_changes","prior_changes",
                        "why_preference"], axis=1)

# Recode Likert responses into numerical code
likert_dict = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, 
               "Agree": 4, "Strongly Agree": 5}
likert_code = {"remote_enjoy":       likert_dict,
               "remote_motivated":   likert_dict,
               "remote_satisfied":   likert_dict,
               "remote_engaging":    likert_dict,
               "remote_distracted":  likert_dict,
               "remote_questions":   likert_dict,
               "prior_enjoy":       likert_dict,
               "prior_motivated":   likert_dict,
               "prior_satisfied":   likert_dict,
               "prior_engaging":    likert_dict,
               "prior_distracted":  likert_dict,
               "prior_questions":   likert_dict}
data.replace(likert_code, inplace=True)

# Only choosing german respondents 

# Acquire data from Germany only 
germany_data = data[data["country"] == "Germany"]
germany_data = germany_data[germany_data["level"] != "High school/A-levels/Gymnasium"]
len(germany_data)

# Acquire data from UK 
uk_data = data[data["country"] == "United Kingdom"] 
uk_data = uk_data[uk_data["level"] == "Undergraduate (studying for associates or bachelors degrees)"] 

# germany_data['primary_mode'].iloc[0:5]
print('Number of respondents in Germany:', len(germany_data))

print('Number of respondents in the UK:', len(uk_data))

# Recode Primary mode entries into shorter things
modes = {"primary_mode":   {"Live classes (ie: Zoom; Google Meet etc.)": "live",
                            "Uploaded or emailed Materials": "upload",
                            "Recorded Lectures": "recorded",
                            "Discussion forums/chats": "chat"}}
germany_data.replace(modes, inplace=True)

germany_data['primary_mode'].iloc[0:5]

# Recode Outside Techniques
techniques = {"techniques_outside":   {"Video Content": "video_content",
                                       "Video Lectures": "video_lectures",
                                       "Discussion/Chat Forums": "chat",
                                       "Assignments/un-proctored exams": "assignments",
                                       "Email Q&A with instructors": "email",
                                       "Posted readings/study material": "readings",
                                       "Live office hours": "office_hours"}}
germany_data.replace(techniques, inplace=True)

# Defining Active and Passive methods of learning 

germany_data["active"] = germany_data["methods_used"].str.contains(pat='In-class assessments/quizzes|Small group activities|Whole class discussion/debate|Q&A with students\' questions|Classroom chat')
# Active methods: 
# - In-class assessments/quizzes 
# - Small group activities
# - Class discussion/debate 
# - Q&A with students 
# - Classroom chats 

germany_data["passive"] = germany_data["methods_used"].str.contains(pat='Lecture/presentation|Explanation using Diagrams/White Boards/other media')
# Passive methods: 
# - Lecture Presentation 
# - Explanation using diagrams 
# - White boards
# - Other media 

germany_data["fully_active"] = germany_data["methods_used"].str.contains(pat='In-class assessments/quizzes|Small group activities|Whole class discussion/debate')
# Fully Active methods:
# - In-class assessments
# - Small group activities 
# - Whole class discussion/debats 

germany_data["partly_active"] = germany_data["methods_used"].str.contains(pat='Q&A with students\' questions|Classroom chat')
# Partly Active methods:
# - Q&A with students/questions 
# - Classroom chats 

    
### Create frames for undergraduates learning synchronously and asynchronously
ug_sync = germany_data[germany_data.primary_mode.eq("live")]
ug_async = germany_data[germany_data["primary_mode"] != "live"]

### Create frames for active / passive / mixed methods
ug_active_only = germany_data[germany_data.active & (germany_data.passive==False)]
ug_passive_only = germany_data[(germany_data.active==False) & (germany_data.passive.eq(True))]
ug_act_pass_mix = germany_data[germany_data.active & germany_data.passive]

ug_fully_active = germany_data[germany_data.fully_active.eq(True)]
ug_partly_active = germany_data[germany_data.partly_active.eq(True)]
ug_passive = germany_data[germany_data.passive.eq(True)]

# Create frames to categorize students by the *most* active technique they list 
ug_f_act = germany_data[germany_data.fully_active.eq(True)]
ug_p_act = germany_data[germany_data.fully_active.eq(False) & germany_data.partly_active.eq(True)]
ug_pass = germany_data[germany_data.fully_active.eq(False) & germany_data.partly_active.eq(False) & germany_data.passive.eq(True)]


# print(len(ug_active_only), len(ug_passive_only), len(ug_act_pass_mix))

### Function for comparing active vs passive sample sizes 
def active_vs_passive(data):
    active_only = data[data.active & (data.passive==False)]
    passive_only = data[(data.active==False)  & (data.passive.eq(True))]
    mixed = data[data.active & data.passive]
    none = data[(data.active == False) & (data.passive == False)]

    return ["active only:",len(active_only)], ["passive only:", len(passive_only)], ["mixed:", len(mixed)], ['none:', len(none)]

print(len(germany_data))
active_vs_passive(germany_data)

germany_data.iloc[0:3,9:-3]

# len(germany_data["active"])
len(germany_data)
# len(ug_act_pass_mix)

germany_data.columns
### ACTIVE VS PASSIVE ### 

actcount = np.zeros((6,5))
pascount = np.zeros((6,5))

for i in range (6):
    for j in range (5):
        actcount[i,j] = ug_active_only[remote_survey_q[i]].value_counts(normalize=True)[j+1] 
        pascount[i,j] = ug_passive_only[remote_survey_q[i]].value_counts(normalize=True)[j+1]

for q in range (6):
    ind = np.arange(5) 
    width = 0.35       
    plt.bar(ind, actcount[q], width, label='Active', color = "brown")
    plt.bar(ind + width, pascount[q], width, label='Passive', color = "gray")

    plt.ylabel('Fraction of Responses')
    plt.title(remote_q_long[q])

    plt.xticks(ind + width / 2, ('Strongly\nDisagree', 'Disagree',
                                 'Neutral', 'Agree', 'Strongly\nAgree'))
    plt.legend(loc='best')
    plt.show()
    print (stats.mannwhitneyu(ug_active_only[remote_survey_q[q]],ug_passive_only[remote_survey_q[q]]))
    print ("Active Only ","{:.3}".format(ug_active_only[remote_survey_q[q]].mean()))
    print ("Passive Only","{:.3}".format(ug_passive_only[remote_survey_q[q]].mean()))
    print ("Cohen's d =","{:.2}".format(cohens_d(ug_active_only[remote_survey_q[q]],ug_passive_only[remote_survey_q[q]])))












### ACTIVE VS MIXED ### 

actcount = np.zeros((6,5))
mixcount = np.zeros((6,5))

for i in range (6):
    for j in range (5):
        actcount[i,j] = ug_active_only[remote_survey_q[i]].value_counts(normalize=True)[j+1] 
        mixcount[i,j] = ug_act_pass_mix[remote_survey_q[i]].value_counts(normalize=True)[j+1]

for q in range (6):
    ind = np.arange(5) 
    width = 0.35       
    plt.bar(ind, actcount[q], width, label='Active')
    plt.bar(ind + width, mixcount[q], width, label='Mixed')

    plt.ylabel('Fraction of Responses')
    plt.title(remote_q_long[q])

    plt.xticks(ind + width / 2, ('Strongly\nDisagree', 'Disagree',
                                 'Neutral', 'Agree', 'Strongly\nAgree'))
    plt.legend(loc='best')
    plt.show()
    print (stats.mannwhitneyu(ug_active_only[remote_survey_q[q]],ug_act_pass_mix[remote_survey_q[q]]))
    print ("Active Only ","{:.3}".format(ug_active_only[remote_survey_q[q]].mean()))
    print ("Mixed","{:.3}".format(ug_act_pass_mix[remote_survey_q[q]].mean()))
    print ("Cohen's d =","{:.2}".format(cohens_d(ug_active_only[remote_survey_q[q]],ug_act_pass_mix[remote_survey_q[q]])))












### PASSIVE VS MIXED ### 
pascount = np.zeros((6,5))
mixcount = np.zeros((6,5))

for i in range (6):
    for j in range (5):
        pascount[i,j] = ug_passive_only[remote_survey_q[i]].value_counts(normalize=True)[j+1] 
        mixcount[i,j] = ug_act_pass_mix[remote_survey_q[i]].value_counts(normalize=True)[j+1]

for q in range (6):
    ind = np.arange(5) 
    width = 0.35       
    plt.bar(ind, pascount[q], width, label='Passive', color = 'gray')
    plt.bar(ind + width, mixcount[q], width, label='Mixed', color = 'brown')

    plt.ylabel('Fraction of Respondents')
    plt.title(remote_q_long[q])

    plt.xticks(ind + width / 2, ('Strongly\nDisagree', 'Disagree',
                                 'Neutral', 'Agree', 'Strongly\nAgree'))
    plt.legend(loc='best')
    plt.show()
    print (stats.mannwhitneyu(ug_passive_only[remote_survey_q[q]],ug_act_pass_mix[remote_survey_q[q]]))
    print ("Passive Only ","{:.3}".format(ug_passive_only[remote_survey_q[q]].mean()))
    print ("Mixed","{:.3}".format(ug_act_pass_mix[remote_survey_q[q]].mean()))
    print ("Cohen's d =","{:.2}".format(cohens_d(ug_passive_only[remote_survey_q[q]],ug_act_pass_mix[remote_survey_q[q]])))








factcount = np.zeros(dim)
pactcount = np.zeros(dim)

for i in range (6):
    for j in range (5):
        factcount[i,j] = ug_f_act[remote_survey_q[i]].value_counts(normalize=True)[j+1]
#         print(factcount)
#         passcount[i,j] = ug_passive_only[remote_survey_q[i]].value_counts(normalize=True)[j+1] 
#         pactcount[i,j] = ug_p_act[remote_survey_q[i]].value_counts(normalize=True)[j+1]

print(ug_f_act[remote_survey_q[5]].value_counts(normalize=True))
print(ug_p_act[remote_survey_q[5]].value_counts(normalize=True))


for q in range (6):
    ind = np.arange(5) 
    width = 0.35       
    plt.bar(ind, factcount[q], width, label='Fully Active')
    plt.bar(ind + width, pactcount[q], width, label='Partly Active')

    plt.ylabel('Fraction of Responses')
    plt.title(remote_q_long[q])

    plt.xticks(ind + width / 2, ('Strongly\nDisagree', 'Disagree',
                                 'Neutral', 'Agree', 'Strongly\nAgree'))
    plt.legend(loc='best')
    plt.show()
    print (stats.mannwhitneyu(ug_f_act[remote_survey_q[q]],ug_passive_only[remote_survey_q[q]]))
    print ("Fully Active ","{:.3}".format(ug_f_act[remote_survey_q[q]].mean()))
    print ("Partly Active","{:.3}".format(ug_p_act[remote_survey_q[q]].mean()))
    print ("Cohen's d =","{:.2}".format(cohens_d(ug_f_act[remote_survey_q[q]],ug_passive_only[remote_survey_q[q]])))
