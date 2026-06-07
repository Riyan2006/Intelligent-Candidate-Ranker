**Redrob Behavioral Signals — Reference**

This document explains the 23 behavioral signals embedded in each candidate's redrob\_signals object, how they relate to candidate quality, and how they're constructed in the synthetic dataset.

# **What are Redrob signals?**
In a real recruiting platform, candidates generate observable behavior beyond what they list in their profile:

- Do they actually respond to recruiter messages?
- Have they logged in recently?
- Did they complete the assessments they started?
- Are recruiters saving their profile?
- Have they completed previous interview cycles?

These behavioral signals are often **more predictive** of whether a candidate can actually be hired than their static profile. A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available.

This dataset includes these signals so that ranking systems can incorporate them as a multiplier or modifier on top of skill-match scoring.

# **The 23 signals**

|**#**|**Signal**|**Range / type**|**What it measures**|
| :- | :- | :- | :- |
|1|profile\_completeness\_score|0-100|How much of the profile they've filled in|
|2|signup\_date|date string|When they signed up on Redrob|
|3|last\_active\_date|date string|When they last logged in|
|4|open\_to\_work\_flag|bool|Have they marked themselves available|
|5|profile\_views\_received\_30d|integer >= 0|How often their profile has been viewed by recruiters in last 30 days|
|6|applications\_submitted\_30d|integer >= 0|How many roles they've applied to recently|
|7|recruiter\_response\_rate|0\.0-1.0|What fraction of recruiter messages they reply to|
|8|avg\_response\_time\_hours|number >= 0|Median time to respond to a recruiter message|
|9|skill\_assessment\_scores|dict[str, 0-100]|Per-skill Redrob assessment scores|
|10|connection\_count|integer >= 0|Number of Redrob connections|
|11|endorsements\_received|integer >= 0|Total skill endorsements received|
|12|notice\_period\_days|0-180|Their stated notice period|
|13|expected\_salary\_range\_inr\_lpa.min / .max|number >= 0|Salary expectations in INR lakhs per annum|
|14|preferred\_work\_mode|onsite/hybrid/remote/flexible|Their stated work-mode preference|
|15|willing\_to\_relocate|bool|Will they relocate if needed|
|16|github\_activity\_score|-1 to 100|GitHub commits/contributions score (-1 if no GitHub linked)|
|17|search\_appearance\_30d|integer >= 0|How often they show up in recruiter searches|
|18|saved\_by\_recruiters\_30d|integer >= 0|How many recruiters bookmarked them in last 30 days|
|19|interview\_completion\_rate|0\.0-1.0|What fraction of interviews they've actually attended|
|20|offer\_acceptance\_rate|-1 to 1.0|What fraction of offers they accepted (-1 if no prior offers)|
|21|verified\_email|bool|Whether their email address is verified|
|22|verified\_phone|bool|Whether their phone number is verified|
|23|linkedin\_connected|bool|Whether their LinkedIn account is connected|

