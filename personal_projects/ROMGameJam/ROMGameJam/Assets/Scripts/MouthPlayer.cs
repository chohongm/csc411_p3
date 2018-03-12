using UnityEngine;
using System.Collections;
using UnityEngine.UI;


public class MouthPlayer : MonoBehaviour {

    public int scoreValue;
    private Done_GameController gameController;
    public float damageValue;

    //public GameObject shot;
    //public Transform shotSpawn;
    public float biteRate;
    private float cooldown;
    private bool entered;
    private Transform prey;
    public GameObject player;

    void Start() {
        cooldown = 0;
    }

    // Update is called once per frame
    void Update() {
        if (cooldown > 0) {
            cooldown -= Time.deltaTime;
        } else {
            if (Input.GetKey("space")) {
                bite();
                cooldown = biteRate;
                //GetComponent<AudioSource>().Play();
            }
        }
    }

    void bite() {
        //play bite animation
        player.GetComponent<Animator>().SetTrigger("attack");

        // if enemy at mouth, bite it!
        if (prey) {
            damageEnemy(prey);
        }
    }

    // attack the enemy in contact.
    void OnTriggerEnter2D(Collider2D other) {
        if (other.gameObject.tag == "Creature") {
            prey = other.transform;
            //gameController.GameOver();
        }
        // add to the score.
        //gameController.AddScore(scoreValue);
    }

    void OnTriggerExit2D(Collider2D other) {
        prey = null;
    }

    void damageEnemy(Transform enemy) {
        Debug.Log("bite!" + Time.time);
        enemy.GetComponent<HealthBar>().damageDisplay(damageValue);
    }
}
