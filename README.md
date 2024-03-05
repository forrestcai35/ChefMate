
<a name="readme-top"></a>




[![LinkedIn][linkedin-shield]][linkedin-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/forrestcai35/ChefMate">
    <img src="Sprites/ChefMateIcon.ico" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">ChefMate</h3>

  <p align="center">
    ChefMate the AI powered culinary assistant.
    <br />
    <a href="https://github.com/forrestcai35/ChefMate/issues">Report Bug</a>
    ·
    <a href="https://github.com/forrestcai35/ChefMate/pulls">Request Feature</a>
  </p>
</div>




<!-- ABOUT THE PROJECT -->
## About The Project
ChefMate is a chatbot-style assistant with integrated recipe storage capabilities that will adapt to your culinary prefernces as you use it!



### Built With

[![Python][python-shield]][python-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![OpenAI][openai-shield]][openai-url]
[![MongoDB][mongodb-shield]][mongodb-url]
[![pandas][pandas-shield]][pandas-url]
[![scikit-learn][scikit-shield]][scikit-url]



<!-- GETTING STARTED -->
## Getting Started

This is how to get ChefMate up and running on your system.

### Prerequisites

Make sure you have Python 3.+ installed, installation can be found at [https://www.python.org/](https://www.python.org/)

### Installation

1. Get a free API Key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Clone the repo
   ```sh
   git clone https://github.com/forrestcai35/ChefMate
   ```
3. Create a virtual environment
   ```sh
      * python3 -m venv .venv
      * source .venv/bin/activate
   ```
4. Install packages
   ```sh
   pip install -r requirements.txt
   ```
5. Create a `.env`file

6. Enter your API in `.env`
   ```js
   OPENAI_API_KEY = 'ENTER YOUR API KEY';
   ```
7. (OPTIONAL) Run this to build.exe file
   ```sh
   pyinstaller --onefile --icon = ChefMateIcon.ico --noconsole App.py
   ```






<!-- USAGE EXAMPLES -->
## Usage


Currently you can use ChefMate to store recipes from [allrecipes.com](allrecipes.com) and [foodnetwork.com](foodnetwork.com). 

ChefMate is also able to create its own unique recipes which you can add to the recipe book by prompting "add recipe".




<!-- CONTRIBUTING -->
## Contributing



If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Any contributions you make are **greatly appreciated**. 
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request





<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.




<!-- CONTACT -->
## Contact

Email: forrestcai35@gmail.com

Project Link: [https://github.com/forrestcai35/ChefMate](https://github.com/forrestcai35/ChefMate)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[license-shield]: https://img.shields.io/badge/MIT-red?style=for-the-badge&label=LICENSE
[license-url]: https://github.com/forrestcai35/ChefMate/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/forrestcai

[python-shield]: https://img.shields.io/badge/Python-%233776AB?style=for-the-badge&logo=Python&labelColor=black
[python-url]: https://python.org

[pytorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=PyTorch&labelColor=black
[pytorch-url]: https://pytorch.org

[openai-shield]: https://img.shields.io/badge/OpenAI-%23412991?style=for-the-badge&logo=OpenAI&labelColor=black
[openai-url]: https://openai.com

[mongodb-shield]: https://img.shields.io/badge/MongoDB-%2347A248?style=for-the-badge&logo=MongoDB&labelColor=black
[mongodb-url]: https://mongodb.com

[pandas-shield]: https://img.shields.io/badge/pandas-%23150458?style=for-the-badge&logo=pandas&labelColor=black
[pandas-url]: https://pandas.pydata.org/


[scikit-shield]: https://img.shields.io/badge/scikit-%23F7931E?style=for-the-badge&logo=scikit-learn&labelColor=black
[scikit-url]: https://scikit-learn.org/stable/