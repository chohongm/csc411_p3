using UnityEngine;
using System.Collections;
using UnityEngine.UI;


public class PlayerStats : MonoBehaviour {

    private Transform healthBar;
    private Image healthBarImg;
    private Transform expBar;
    private Image expBarImg;
    private float timer;
    public GameObject explosion;
    public GameObject playerExplosion;
    public Transform player;
    public Sprite[] evolutionSprites;
    private int level;

    // Use this for initialization
    void Start () {
        expBar = transform.GetChild(1);
        expBarImg = expBar.GetChild(0).GetComponent<Image>();
        healthBar = transform.GetChild(0);
        healthBarImg = healthBar.GetChild(0).GetComponent<Image>();
        //Debug.Log(transform.GetChild(1));
        level = 0;
    }
	
	// Update is called once per frame
	void Update () {
	
	}

    public void damageDisplay(float damage) {
        Debug.Log("Called!");
        healthBarImg.fillAmount -= damage;
        // if player health <= 0, destroy the player.
        if (healthBarImg.fillAmount <= 0) {
            // explode this creature
            Instantiate(explosion, player.position, player.rotation);
            Destroy(player.gameObject);
        }
    }

    public void rewardExp(float exp) {
        Debug.Log(exp);
        float tempExp = expBarImg.fillAmount + exp;
        Debug.Log(tempExp);

        if (tempExp >= 1f) {
            if (level < evolutionSprites.Length) {
                expBarImg.fillAmount = tempExp - 1;
                evolutionize();
            } else {
                expBarImg.fillAmount = 1;
            }
        }
    }

    void evolutionize() {
        level++;
        player.GetComponent<SpriteRenderer>().sprite = evolutionSprites[level - 1];
    }
}
