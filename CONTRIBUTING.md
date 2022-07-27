
# Guide to Pull Requests

[Original Document](https://docs.google.com/document/d/1aLAcnlmGrdkQs1MPBond2_KkXAZZrOoEdsZuDez7QEo/edit#heading=h.t6b5v7g2gw3o)


### Guidelines



* To merge any pull request, a CI engine would test all the tests. No pull requests can be merged that don’t pass these tests.
* Create small pull requests.
* Write useful descriptions and titles.
* Add comments on your pull request to help guide the reviewer.
* If you show use cases, it makes it easier for other developers to use your functionality.
* [Developers] Go through this: [Google Engg Practices for developers.](https://github.com/google/eng-practices/blob/master/review/developer/index.md)
* [Reviewers] Go through this: [Google Engg Practices for reviewers.](https://github.com/google/eng-practices/blob/master/review/reviewer/index.md)
* _Documentation style fixing._


### Process



* A pull request	should ideally be linked to a well described issue. The description should include:
    * What issue your PR is related to.
    * What change your PR adds.
    * How you tested your change.
    * Anything you want reviewers to scrutinize.
    * Any other information you think reviewers need.
* All pull requests implementing a new functionality should be accompanied with basic tests.
* After raising a pull request, the developer should ask at least 2 reviewers to provide comments offline in their own time. If there’s a need for clarifications, the reviewers are advised to ask comments on the pull request itself. (So we have documentation for any changes).
* The review comments should be worked on and the developer should ask for another review.
* Once this iterative process has been completed, merge the changes to the main branch.
* The changes pushed to the main branch would run unit and integration tests within the tests folder, if passed, the branch is merged. ([CI](https://www.atlassian.com/continuous-delivery/continuous-integration#:~:text=Continuous%20integration%20(CI)%20is%20the,builds%20and%20tests%20then%20run.))
* Some kind of broadcast to update others on the team ([Github + Slack](https://slack.github.com/)).


### References



* [https://developers.google.com/blockly/guides/modify/contribute/write_a_good_pr](https://developers.google.com/blockly/guides/modify/contribute/write_a_good_pr)
* [https://www.atlassian.com/blog/git/written-unwritten-guide-pull-requests](https://www.atlassian.com/blog/git/written-unwritten-guide-pull-requests)
* [https://github.com/google/eng-practices/blob/master/review/developer/index.md](https://github.com/google/eng-practices/blob/master/review/developer/index.md)




### Suggestions



* Setting Reviewer domains
* Every issue through PR
* One Senior in every review
* Fixing documentation style
* Fixing Issue creation style
* Create a sample Issue template
* Create a sample PR template
