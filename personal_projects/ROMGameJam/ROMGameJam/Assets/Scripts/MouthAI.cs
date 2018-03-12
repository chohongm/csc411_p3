using UnityEngine;
using System.Collections;
using UnityEngine.UI;


public class MouthAI : MonoBehaviour {

    public int scoreValue;
    private Done_GameController gameController;
    public float damageValue;
    private Transform playerStatsCanvas;

    //public GameObject shot;
    //public Transform shotSpawn;
    public float biteRate;
    private float timer;
    private bool entered;
    private Transform player;

    void Start() {
        entered = false;
        playerStatsCanvas = GameObject.Find("PlayerStatsCanvas").transform;
    }
	
	// Update is called once per frame
	void Update () {
        if (entered) {
            timer += Time.deltaTime;
            if (timer >= biteRate) {
                timer = 0;
                damagePlayer(player);
            }
        }
	}

    // attack the enemy in contact.
    void OnTriggerEnter2D (Collider2D other) {
        // in contact with player,
        if (other.tag == "Player") {
            timer = biteRate - 1;
            entered = true;
            if (other.gameObject.tag == "Player") {
                player = other.transform;
                //gameController.GameOver();
            }
        }
        // add to the score.
        //gameController.AddScore(scoreValue);
    }

    void OnTriggerExit2D(Collider2D other) {
        entered = false;
    }

    void damagePlayer(Transform player) {
        Debug.Log("dmg player! by: " + transform.parent.name);
        playerStatsCanvas.GetComponent<PlayerStats>().damageDisplay(damageValue);
    }
}
