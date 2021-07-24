from bareSnake import snakeGame

a = snakeGame()

loop = True

while loop:
    print(a.direction)
    print(a.pos[0])
    print(a.pos[1])
    print(a.gameOver)
    print(a.toString())
    print("east")
    print(a.propDirect([10, 0]))
    print("south")
    print(a.propDirect([0, 10]))
    print("west")
    print(a.propDirect([-10, 0]))
    print("north")
    print(a.propDirect([0, -10]))

    print("south east")
    print(a.propDirect([10, 10]))
    print("south west")
    print(a.propDirect([-10, 10]))
    print("north east")
    print(a.propDirect([10, -10]))
    print("north west")
    print(a.propDirect([-10, -10]))

    print(a.nearWall())
    print(a.pos)
    print(a.sight)

    x = input("enter a command \n")
    if x == "a":
        print("\n\nLook here")
        print(a.movedTowardsFood('left'))
        print(a.movedFromWall("left"))
        print(a.munchMove("left"))
        a.move_left()
        #print(a.toString())
    if x == "s":
        print("\n\nLook here")
        print(a.movedTowardsFood('down'))
        print(a.movedFromWall("down"))
        print(a.munchMove("down"))
        a.move_down()
        #print(a.toString())
    if x == "d":
        print("\n\nLook here")
        print(a.movedTowardsFood('right'))
        print(a.movedFromWall("right"))
        print(a.munchMove("right"))
        a.move_right()
        #print(a.toString())
    if x == "w":
        print("\n\nLook here")
        print(a.movedTowardsFood('up'))
        print(a.movedFromWall("up"))
        print(a.munchMove("up"))
        a.move_up()
        #print(a.toString())
    if x == "h":
        loop = False
    if a.checkGame() == False:
        print("you died")
        loop = False