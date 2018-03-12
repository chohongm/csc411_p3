using UnityEngine;
using System.Collections;
using UnityEngine.SceneManagement;

public class GameController : MonoBehaviour
{
    private bool creatureSelected = false; // restricts player movement until they select a creature


    public GUIText scoreText;
    public GUIText restartText;
    public GUIText gameOverText;

    private bool gameOver;
    private bool restart;
    private int score;

    public Transform player;

    void Start() {
        gameOver = false;
        restart = false;
        restartText.text = "";
        gameOverText.text = "";
        score = 0;
        UpdateScore();
    }

    void Update() {
        if (restart) {
            if (Input.GetKeyDown(KeyCode.R)) {
                Application.LoadLevel(Application.loadedLevel);
            }
        }
        if (player == null) {
            GameOver();
        }
    }

    public void AddScore(int newScoreValue) {
        score += newScoreValue;
        UpdateScore();
    }

    void UpdateScore() {
        scoreText.text = "Score: " + score;
    }

    public void GameOver() {
        gameOverText.text = "Game Over!";
        gameOver = true;
        restartText.text = "Press 'R' for Restart";
        restart = true;
    }

    // Sets creatureSelected to val
    public void SetCreatureSelected(bool val)
    {
        creatureSelected = val;
    }

    public bool GetCreatureSelected()
    {
        return creatureSelected;
    }

}