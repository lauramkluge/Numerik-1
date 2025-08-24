import task1
import task2
import task3
import task4


benutzereingabe = 5
while benutzereingabe != 0:
    benutzereingabe = input("Die Loesung welcher Aufgabe soll gezeigt werden? (moegliche Werte: 1,2,3,4) Wähle 0 zum "
                            "verlassen des Programms.")
    benutzereingabe = int(benutzereingabe)
    if benutzereingabe == 1:
        print(
            'Die maximale Abweichung vom Mittelwert der numpy Funktion beträgt ' + str(task1.hundretcomparisions()[0]) +
            ' und die maximale Abweichung vom Median der numpy Funktion beträgt ' + str(
                task1.hundretcomparisions()[1]) + ' .')
    elif benutzereingabe == 2:
        task2.showfilteredave(2, 1)
    elif benutzereingabe == 3:
        task3.showfilteredmed(2, 1)
    elif benutzereingabe == 4:
        task4.showfilteredbil(2, 75, 1)
    elif benutzereingabe == 0:
        print('Tschüss!')
    else:
        print('Falsche Eingabe!')
