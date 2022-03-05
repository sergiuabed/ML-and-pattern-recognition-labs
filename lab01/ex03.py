import sys

class Birth:
    def __init__(self, name, surname, birthplace, birthdate):
        self.name=name
        self.surname=surname
        self.birthplace=birthplace
        self.birthdate=birthdate

def nrBirthsInCities(birthsList):
    cityBirth={}    #dictionary
    for birth in birthsList:
        if not birth.birthplace in cityBirth:
            cityBirth[birth.birthplace]=1
        else:
            cityBirth[birth.birthplace]+=1

    return cityBirth

def nrBirthsInMonth(birthsList):
    monthBirth={}
    for birth in birthsList:
        day, month, year=birth.birthdate.split('/')
        monthYear=month+'/'+year
        if not monthYear in monthBirth:
            monthBirth[monthYear]=1
        else:
            monthBirth[monthYear]+=1
    return monthBirth

def avgBirthsCity(birthsList):
    cityBirth=nrBirthsInCities(birthsList)
    count=0
    nr=0
    for city in cityBirth:
        nr+=cityBirth[city]
        count+=1
    
    return nr/count

if __name__=='__main__':
    birthsList=[]

    with open(sys.argv[1], 'r') as f:
        for line in f:
            name, surname, birthplace, birthdate=line.strip().split()
            birth=Birth(name, surname, birthplace, birthdate)
            birthsList.append(birth)

    cityBirth=nrBirthsInCities(birthsList)

    print('Births per city')
    for birth in cityBirth:
        print('%s:  %d' % (birth, cityBirth[birth]))

    monthBirth=nrBirthsInMonth(birthsList)

    print('\nBirths per month')
    for birth in monthBirth:
        print('%s:  %d' % (birth, monthBirth[birth]))

    print('Average number of births:    %.2f' % (avgBirthsCity(birthsList)))
