import sys

class Record:
    def __init__(self, busID, routeNr, x, y, time):
        self.busID=busID
        self.routeNr=routeNr
        self.x=x
        self.y=y
        self.time=time

    def __str__(self):
        return '%d %d %d %d %d' % (self.busID, self.routeNr, self.x, self.y, self.time)


def traveledDistance(recordList, busID):
    dist=0
    startPoint=0
    endPoint=0
    for record in recordList:
        if record.busID==busID:
            startPoint=endPoint
            endPoint=record

            if startPoint==0:
                continue

            dist+=((record.x-startPoint.x)**2 + (record.y-startPoint.y)**2)**0.5

    return dist

def timeOnRoute(recordList, busID):
    totTime=0
    startPoint=0
    endPoint=0
    for record in recordList:
        if record.busID==busID:
            startPoint=endPoint
            endPoint=record

            if startPoint==0:
                continue

            totTime+=endPoint.time-startPoint.time
    return totTime

def avgSpeed(recordList, routeNr):
    dist=0
    time=0 
    listBuses=set()    #set of buses on route routeNr

    for record in recordList:
        if record.routeNr==routeNr:
            listBuses.add(record.busID)

    for busID in listBuses:
        dist+=traveledDistance(recordList, busID)
        time+=timeOnRoute(recordList, busID)

    avg=dist/time

    return avg

if __name__=='__main__':
    recordList = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            busID,routeNr,x,y,time=line.strip().split()
            rec=Record(int(busID), int(routeNr), int(x), int(y), int(time))
            recordList.append(rec)

    if sys.argv[2]=='-b':
        dist=traveledDistance(recordList, int(sys.argv[3]))
        print('%d - Total Distance: %f' % (int(sys.argv[3]), dist))
    elif sys.argv[2]=='-l':
        avg_speed=avgSpeed(recordList, int(sys.argv[3]))
        print('%d - Avg Speed: %f' % (int(sys.argv[3]), avg_speed))


