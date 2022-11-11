from typing import List

class Passenger:
    pass
# Association
class Passengers:
    airplane = None
    passengers: List[Passenger] = [] # This makes passengers an aggregation of passengers
    
class Airplane:
    passengers = None
    
    
# Multiplicty
class LimitedAirplane():
    capacity = 300
    passengers = []
    def add_passenger(self, p: Passenger):
        if len(self.passengers) < self.capacity:
            self.passengers.append(p)

class Wing:
    pass

class Body:
    pass

# Composition
class FullAirplane:
    left_wing:Wing
    right_wing:Wing
    body:Body


if __name__ == '__main__':
    # Association
    p = Passengers()
    a = Airplane()
    p.airplane = a
    a.passengers = p
    
    
