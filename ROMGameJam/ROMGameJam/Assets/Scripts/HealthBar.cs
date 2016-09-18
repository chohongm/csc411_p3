using UnityEngine;
using System.Collections;
using UnityEngine.UI;

/// <summary>
/// WORLD SPACE CANVAS FACES PLAYER AT ALL TIMES
/// </summary>
public class HealthBar : MonoBehaviour
{
    private Transform progressCanvas;
    private Transform progressBar;
    private Image progressBarImg;
    private float timer;
    public GameObject explosion;
    public GameObject playerExplosion;
    private Transform playerStatsCanvas;
    public float exp;
    public float dmg;
    public float health;


    void Start()
    {
        playerStatsCanvas = GameObject.Find("PlayerStatsCanvas").transform;
        progressCanvas = transform.Find("ProgressCanvas");
        /*
        if (!progressCanvas.gameObject.activeSelf) {
            progressCanvas.gameObject.SetActive(true);
        }
        */
        progressBar = progressCanvas.Find("ProgressBG");
        progressBarImg = progressBar.Find("Progress").GetComponent<Image>();
        progressCanvas.gameObject.SetActive(false);
        timer = 0;
    }

    void Update()
    {
        // if canvas active, keep it 0 degrees.
        if (progressCanvas.gameObject.activeSelf) {
            progressCanvas.rotation = Quaternion.Euler(0, 0, 0);
            timer += Time.deltaTime;
        }
        // turn off canvas after 3 sec display
        if (timer >= 3) {
            progressCanvas.gameObject.SetActive(false);
            timer = 0;
        }
    }

    public void damageDisplay(float damage) {
        timer = 0;
        if (!progressCanvas.gameObject.activeSelf) {
            progressCanvas.gameObject.SetActive(true);
        }
        // if this.mass <= player.mass, do nothing (just bump into each other)
        // if otherwise, dmg the player based on mass difference.
        progressBarImg.fillAmount -= damage;
        //Debug.Log(progressBarImg.fillAmount);
        // if player health <= 0, destroy the player.
        if (progressBarImg.fillAmount <= 0) {
            // explode this creature
            Instantiate(explosion, transform.position, transform.rotation);
            Destroy(gameObject);
            playerStatsCanvas.GetComponent<PlayerStats>().rewardExp(exp);
        }
    }
}