import sys

gymnastScore={}
countryScore={}
with open(sys.argv[1], 'r') as f:
    for line in f:
        name, surname, country, *scores=line.strip().split(" ")
        sum=0
        scores=sorted(scores)
        scores=scores[1:-1] #extract best and worst scores (in case of duplicates of best and worst scores, remove just one copy of each)
        for score in scores:
            sum+=float(score)
        
        fullName=name+' '+surname
        gymnastScore[fullName]=sum

        if not country in countryScore:
            countryScore[country]=0
        countryScore[country]=countryScore[country]+sum

gymnastScore_list=list(gymnastScore.items())    #convert dictionary to list of couples (name_gymnasts, sum_scores)
topGymnasts=sorted(gymnastScore_list, key=lambda e: e[1], reverse=True)[0:3]    #extract top 3 gymnasts
print("final ranking:")
for t in topGymnasts:
    print("%s   Score: %f" % (t[0], t[1]))

print("\nBest Country:")
countryScore_list=list(countryScore.items())    #convert dictionary to list of couples (country, sum_scores)
topCountry=sorted(countryScore_list, key=lambda e: e[1], reverse=True)[0]       #extract top country
print(topCountry)






