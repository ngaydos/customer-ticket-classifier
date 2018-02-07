# Classifying Customer Tickets using Flowroute Support Tickets

![image flowroute logo](Flowroute_logo_2017.png)

## Confidentiality
As part of this project, I was given access to private customer ticket data from Flowroute. If you need a demonstration or would like to discuss specifics, please feel free to reach out at ngaydos{@}gmail.com


## Business Understanding
Modern tech companies find themselves with new and unique challenges when it comes to approaching customer support. They may have a wide customer base and very specialized product support. This can waste time, cause incorrect responses to customers and even team tension or ownership disputes as tickets for different teams come in through the same channels.


##Data Understanding
The data used in this project was ~8000 customer tickets provided by Flowroute Inc., a small VOIP technology company located in Seattle. The data included the subject line, body line and ticket final classification as well as various tags and customer information.

Much of the customer information as well as the tagging system were found to contain data leakage and thus were not included in the final model. The final team assignment was determined to the target value for the model. Of the final team assignments, 5 tickets were determined to be outlier tickets (assigned to teams that did not actually exist, or misassigned) and were removed from the training dataset. Approximately 87% of the remaining tickets were assigned to the general support team and the remaining tickets were assigned to the number portability team.


##Data Preparation
Due to the leaky nature of the other data, the only columns that were used in the predictions for the model were the case subject and the case body. Other columns were either not predictive or were directly referential to things that would be unavailable in a newly received customer ticket.

The text from the body and case subject was then made lowercase and all punctuations was removed. Stopwords (taken from common words at http://www.textfixer.com/resources/common-english-words.txt ) were removed, and the words were reduced to their base form. 


##Modeling