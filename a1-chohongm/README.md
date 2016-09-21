## Learning goals

 * Get you familiar with the tools we'll be using (Git, GitHub, Travis CI, etc.)
 * Get you used to our individual assignment workflow
 * Refresh your basic Java skills

## Getting Started

 1. [Fork][github-fork] this repo.

     > _Note:_ Your fork is private and is only visible to you, the TAs and the instructors.

 2. Clone the fork to your local machine (using the `git clone` command).
 
 3. Open the project in your IDE.       
    If you are using Eclipse, _File --> Import --> Existing Maven Projects_.


## Your Task

Implement the code to pass the provided unit tests.        
See the marking scheme below for more details.

### Guidance

We (the teaching staff) want you (the students) to become comfortable reading code.     
Therefore, the requirements for this assignment are specified almost entirely in code, using unit tests.

We encourage you to jump into the code, try to understand it and use the discussion board to ask questions.

We suggest that you start by running the tests in [`SetupTest`](src/test/java/edu/toronto/csc301/SetupTest.java) (In Eclipse, _right click -> Run As -> JUnit Test_):

![Screenshot](https://csc301-fall-2016.github.io/img/eclipse-screenshot.png)

These tests specify which classes you need to create, which interfaces they should implement and which constructor(s) each class must have. Once you pass these tests, you can fill in the rest of the implementation based on the other test classes.

_Note:_ This assignment is slightly different than a real _test driven development_ environment.
In most realistic scenarios, you will not start with a full set of test cases. Instead, you will write a test (or a few tests), then write the code to pass the test(s) and repeat the process.



## Deliverables

Your code, submitted as a single, non-conflicting [pull-request][github-pull-requests] from your fork to the handout repo (i.e. the repo you forked).

## Marking Scheme

Your code will be **marked automatically**, according to the following scheme:

 * 100% : Passing all tests (i.e. Get a green light from Travis CI)
 * 50%  : Failing at most 3 tests
 * 0    : Failing more than 3 tests (or not submitting a solution)


## Important Notes

 1. Do not add any collaborators or teams to your fork!

    > Since you are the owner of your fork repo, GitHub allows you to share it with
others. Note that GitHub also allows us (the instructors and TA's) to see if
share your fork with anyone.

  If you share your fork with anyone, you will **automatically get a 0 mark** for this assignment.
  
 2. After you submit your assignment, make sure to check the results of Travis CI.
 
     > If your code passes the tests on your computer, but fails on Travis, then your code is broken.       

    It is **your responsibility** to make sure your code compiles!
  
 3. Do not modify any of given interfaces or testing code!
 
    > If you do, then Travis will no longer be useful, since it will no longer run the same tests as our auto-marker.
    


[github-issues]: https://guides.github.com/features/issues/
[github-guides]: https://guides.github.com/ "GitHub guides"
[github-fork]: https://guides.github.com/activities/forking/ "Guide to GitHub fork"
[github-pull-requests]: https://help.github.com/articles/using-pull-requests/ "Guide to GitHub Pull-Requests"
