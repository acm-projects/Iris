<p align="center">
  <img src="./justina-miles.gif" alt="Idiot Sandwich" width="600"/>
</p>

<h1 align="center">🌸 Iris 🌸</h1>

<p align="center">
Video calls have become the default way people meet, but they're built entirely around spoken language. For someone who is deaf or hard of hearing and doesn't have a hearing interpreter on the call, a Google Meet is functionally unusable. Captions only work one direction, and there's no way to sign back and be understood. Iris is a real-time translation layer for video meetings. It uses computer vision and machine learning to read a user's sign language through their camera and speak it aloud to hearing participants, while simultaneously converting hearing participants' speech into on-screen captions for the signer. Unlike existing sign-recognition tools that require specialized hardware or are built purely for teaching, Iris is designed to sit invisibly inside a call people are already having, turning a one-way captioning feature into an actual two-way conversation.
</p>

---

## MVP ✅
* **Meeting Integration** → Connection to Google Meet
* **Sign Recognition Pipeline** → Gesture classification using the pretrained Pose-TGCN model
* **Speech Output** → ASL converted to Speech
* **Type-To-Voice** → Alternative Type to Speech
* **Signer-Facing Feedback** → Display of the currently detected sign in real time
* **Captions for the Signer** → Captions that show for other people talking

---

## Stretch Goals 💪
* **Phrase Predictions** → Predictive text/phrase suggestions
* **Multiple Languages** → Support sign language dialects beyond ASL (e.g., BSL, LSF)

---

## Tech Stack & Resources 💻

<details>
<summary>📱 Frontend</summary>

* [React Documentation](https://react.dev/learn)
* [Tailwind CSS Documentation](https://tailwindcss.com/docs)
* [Chrome Extensions Documentation](https://developer.chrome.com/docs/extensions/)
* [Chrome Extensions: Get Started Tutorial](https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world)
* [VIDEO: React JS Full Course for Beginners](https://www.youtube.com/watch?v=SqcY0GlETPk)
* [VIDEO: Chrome Extension Development Tutorial for Beginners](https://www.youtube.com/watch?v=0n809nd4Zu4)
* [VIDEO: Tailwind CSS Full Course for Beginners](https://www.youtube.com/watch?v=DenUCuq4G04)

</details>

<details>
<summary>🔐 Auth</summary>

* [Google OAuth with FastAPI](https://developers.google.com/identity/protocols/oauth2)
* [Passport.js](http://www.passportjs.org/)
* [Firebase Authentication](https://firebase.google.com/docs/auth)
* [VIDEO: Code with Me: Learning FastAPI + Google OAuth2](https://www.youtube.com/watch?v=EIZTy4XyXjU)
* [VIDEO: FastAPI Authentication Example With OAuth2, JSON Web Tokens and Tortoise ORM](https://www.youtube.com/watch?v=6hTRw_HK3Ts)
* [VIDEO: Node.js Passport Login System Tutorial](https://www.youtube.com/watch?v=-RCnNyD0L-s)
* [VIDEO: Express JS #15 - Passport.js Authentication](https://www.youtube.com/watch?v=_lZUq39FGv0)

</details>

<details>
<summary>👀 Vision & Recognition</summary>

* [MediaPipe (Hollistic Landmarker)](https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker/python)
* [Sign Classification: Pose-TGCN](https://github.com/dxli94/WLASL/blob/master/README.md)
* [PyTorch + scikit-learn](https://docs.pytorch.org/docs/stable/index.html)

</details>

<details>
<summary>💬 Speech Layer</summary>

* [Speech-to-Text](https://join.elevenlabs.io/agents/v6?utm_source=google&utm_medium=cpc&utm_campaign=us_nonbrandsearch_conversationalai_english&utm_id=22795646787&utm_term=how+to+build+voice+ai+agents&utm_content=conversational_ai_-_voice_ai_agents&gad_source=1&gad_campaignid=22795646787&gbraid=0AAAAA_PU6Fb9mgQFDnfTH_B0gHF7C80RW&gclid=CjwKCAjwtp7VBhBjEiwAJfpV-9GOsq60wrkT_VC4EjzUBjuPIBa1IhkQBesTuZZTo3z8JTdz6yOGjRoCvs8QAvD_BwE)
* [Virtual Audio Cable](https://vb-audio.com/Cable/?utm_source=chatgpt.com)
* [PyTorch + scikit-learn](https://docs.pytorch.org/docs/stable/index.html)
* [OBS Virtual Camera](https://obsproject.com/)

</details>

<details>
<summary>🗄️ Database</summary>

* [Firebase Firestore](https://firebase.google.com/docs/firestore)
* [PostgreSQL Docs](https://www.postgresql.org/docs/)
* [Supabase](https://supabase.com/)
* [AWS S3](https://aws.amazon.com/s3/)
* [VIDEO: Firebase Firestore Tutorial #1 - Introduction (full series)](https://www.youtube.com/watch?v=4d-gIPGzmK4)
* [VIDEO: PostgreSQL Tutorial Full Course 2022](https://www.youtube.com/watch?v=85pG_pDkITY)
* [VIDEO: AWS S3 Tutorial for Beginners 2023 | AWS S3 Overview | Cloud Computing](https://www.youtube.com/watch?v=HAuvciBEIMQ)

</details>

<details>
<summary>🎨 Design</summary>

* [Figma](https://www.figma.com/)
* [LottieFiles](https://lottiefiles.com/)
* [VIDEO: Figma Tutorial for Beginners](https://www.youtube.com/watch?v=ezldKx-jPag&pp=0gcJCfwAo7VqN5tD)

</details>

<details>
<summary>🛠️ Dev Tools</summary>

* Node.js:
Download: [Node.js LTS](https://nodejs.org/en/download/)
Tutorial: [Installing Node.js and npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)

* VS Code:
Download: [Visual Studio Code](https://code.visualstudio.com/download)
Tutorial: [VS Code Tips for React Native](https://code.visualstudio.com/docs/nodejs/reactjs-tutorial)

* Postman (API Testing):
Download: [Postman](https://www.postman.com/downloads/)
Tutorial: [Postman API Testing for Beginners](https://www.guru99.com/postman-tutorial.html)

* Git:
Download: [Git](https://git-scm.com/downloads)
Tutorial: [Git Handbook](https://guides.github.com/introduction/git-handbook/)

* Python:
Download: [Python](https://www.python.org/downloads/)
Tutorial: [FastAPI Quickstart](https://fastapi.tiangolo.com/tutorial/)

</details>

---

## Roadmap 📅

<table>
  <tr>
    <th>Week</th>
    <th>Frontend</th>
    <th>Backend</th>
  </tr>
  <tr>
    <td>1</td>
    <td>
        <ul>
            <li>Decide roles, discuss project plan, design day</li>
            <li>Decide integration approach (browser extension vs. virtual camera/mic) and UI layout</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Set up repo</li>
            <li>Load the pretrained Pose-TGCN (PyTorch) checkpoint and confirm it runs on a static test clip's precomputed keypoints (not webcam yet).</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>2</td>
    <td>
        <ul>
            <li>Build meeting-overlay UI: detected sign display, confidence score bar</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Build the MediaPipe model keypoint remapping/normalization layer, and a rolling buffer that samples frames into the model's expected input shape.</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>3</td>
    <td>
        <ul>
            <li>Build caption display panel for incoming speech</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Integrate a speech-to-text API (e.g., Web Speech API / Google Cloud Speech-to-Text)</li>
            <li>Build sign-boundary segmentation (pause-based or sliding-window) so continuous signing gets split into individual signs before classification.
</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>4</td>
    <td>
        <ul>
            <li>Style the sentence-builder UI (running transcript of assembled signs)</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Build sentence-assembly logic: buffer recognized signs into phrases with basic grammar rules</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>5</td>
    <td>
        <ul>
            <li>Wire TTS audio into UI controls (mute/replay last phrase)</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Integrate TTS engine; route synthesized audio into a virtual audio output device</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>6</td>
    <td>
        <ul>
            <li>Build Google Meet integration UI (join flow, camera/mic device selection)</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Implement virtual camera/mic injection so Iris outputs appear as native call audio/video</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>7</td>
    <td>
        <ul>
            <li>Polish caption/sign overlay positioning and readability</li>
            <li>START PRESENTATION PREP and freeze new features</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Tune model confidence thresholds; add a "low confidence please repeat" prompt</li>
            <li>Tune model confidence thresholds; add a "low confidence please repeat" prompt</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>8</td>
    <td>
        <ul>
            <li>Polish UI colors/fonts/accessibility (contrast, font size for captions)</li>
            <li>Presentation Prep.</li>
        </ul>
    </td>
    <td>
        <ul>
            <li>Fix critical bugs; test full round-trip latency (sign → speech, speech → caption)</li>
            <li>Presentation Prep.</li>
        </ul>
    </td>
  </tr>
  <tr>
    <td>9</td>
    <td colspan="2" align="center">✨ Presentation Prep ✨</td>
  </tr>
</table>

---

## GitHub Cheat Sheet 💬

| Command | Description |
| ------ | ------ |
| **cd <director>** | Change directories over to our repository |
| **git branch** | Lists branches for you |
| **git branch "branch name"** | Makes new branch |
| **git checkout "branch name"** | Switch to branch |
| **git checkout -b "branch name"** | Same as 2 previous commands together |
| **git add .**| Finds all changed files |
| **git commit -m "Testing123"** | Commit with message |
| **git push origin "branch"** | Push to branch |
| **git pull origin "branch"** | Pull updates from a specific branch |
| **git commit hash** (find on GitHub or run `git log --oneline` in the terminal), then run **`git revert <commit-hash> --no-edit`** | Undo a commit that has been pushed |
| **git reset --soft HEAD~** | Undo commit (not pushed) but *keep* the changes |
| **get commit hash** (run `git log --oneline`), then **`git reset --hard <commit-hash>`** | Undo a commit that has not been pushed and *remove* its changes |

## The Team 🎉

<div align="center">
<h2>🎊Developers🎊</h2>
<h3>Bratesh Gupta</h3><br/>
<h3>Julian Castellanos</h3><br/>
<h3>Karina Gonzalez-Martinez</h3><br/>
<h3>Laya Mallina</h3><br/>
<h3>Safa Ahmed</h3><br/>
<h2>🎊Project Manager🎊</h2>
<h3>Hadent Dang</h3><br/>
<h2>🎊Industry Mentor🎊</h2>
<h3>Abis Naqvi</h3><br/>
<div />
