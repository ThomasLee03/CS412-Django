
#file: models.py
# author: Thomas Lee (tlee03@bu.edu), 11/10/2024
from django.db import models

# Create your models here.
class Voter(models.Model):

    '''
    Store/represent the data from one runner at the newton voters.
    Last_name, first_name, streetNumber, streetName, aptNumber, ZIPCode, dob, dor, partyAffiliation, PrecinctNumber, v20state, v21town, v21primary, v22general, v23town, voter_score
    '''
    # identification
    last_name = models.TextField()
    first_name = models.TextField()
    streetNumber = models.IntegerField()
    streetName = models.TextField()
    aptNumber = models.TextField(blank = True) 
    ZIPCode = models.IntegerField()
    dob = models.DateField(blank=True, null=True)
    dor = models.DateField(blank=True, null=True)
    partyAffiliation = models.TextField() 
    PrecinctNumber = models.TextField() 
    v20state = models.TextField()
    v21town = models.TextField()
    v21primary = models.TextField()
    v22general = models.TextField()
    v23town = models.TextField()
    voter_score = models.IntegerField()

    def __str__(self):
        '''Return a string representation of this model instance.'''
        return f'{self.first_name} {self.last_name}({self.partyAffiliation})'
    
def load_data():
    '''Function to load data records from CSV file into Django model instances.'''

    # Delete existing records to prevent duplicates
    Voter.objects.all().delete()
    filename = 'voter_analytics/newton_voters.csv'
    
    with open(filename) as f:
        f.readline()  # Discard headers

        for i, line in enumerate(f, start=1):
            line = line.strip()
            fields = line.split(',')

            try:
             #   print(f"Processing row {i}: {fields}")  # Print the row being processed

                # Retrieve and clean each field, with debug statements for each
                last_name = fields[1].strip()
              #  print(f"last_name: {last_name}")
                
                first_name = fields[2].strip()
              #  print(f"first_name: {first_name}")

                streetNumber = int(fields[3])
             #   print(f"streetNumber: {streetNumber}")
                
                streetName = fields[4].strip()
               # print(f"streetName: {streetName}")

                aptNumber = fields[5].strip() if fields[5] else ""
              #  print(f"aptNumber: {aptNumber}")

                ZIPCode = int(fields[6])
              #  print(f"ZIPCode: {ZIPCode}")

                dob = fields[7].strip() if fields[7] else None
              #  print(f"dob: {dob}")

                dor = fields[8].strip() if fields[8] else None
              #  print(f"dor: {dor}")

                partyAffiliation = fields[9].strip()
               # print(f"partyAffiliation: {partyAffiliation}")

                PrecinctNumber = fields[10].strip()
               # print(f"PrecinctNumber: {PrecinctNumber}")

                v20state = fields[11].strip()
               # print(f"v20state: {v20state}")

                v21town = fields[12].strip()
               # print(f"v21town: {v21town}")

                v21primary = fields[13].strip()
               # print(f"v21primary: {v21primary}")

                v22general = fields[14].strip()
              #  print(f"v22general: {v22general}")

                v23town = fields[15].strip()
              #  print(f"v23town: {v23town}")

                voter_score = int(fields[16].strip())
              #  print(f"voter_score: {voter_score}")

                # Create a new instance of Voter with all processed fields
                voter = Voter(
                    last_name=last_name,
                    first_name=first_name,
                    streetNumber=streetNumber,
                    streetName=streetName,
                    aptNumber=aptNumber,
                    ZIPCode=ZIPCode,
                    dob=dob,
                    dor=dor,
                    partyAffiliation=partyAffiliation,
                    PrecinctNumber=PrecinctNumber,
                    v20state=v20state,
                    v21town=v21town,
                    v21primary=v21primary,
                    v22general=v22general,
                    v23town=v23town,
                    voter_score=voter_score
                )

                voter.save()  # Commit to the database
               # print(f"Created voter: {voter}")

            except Exception as e:
                print(f"Error in row {i}, field: {fields}, due to: {e}")
    print(f'Done. Created {Voter.objects.count()} voter records.')
