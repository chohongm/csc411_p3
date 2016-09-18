using UnityEngine;
using System.Collections;
using UnityEngine.UI;


public class Done_DestroyByContact : MonoBehaviour
{
	public GameObject explosion;
	public GameObject playerExplosion;
	public int scoreValue;
	private Done_GameController gameController;

	void Start ()
	{
		GameObject gameControllerObject = GameObject.FindGameObjectWithTag ("GameController");
		if (gameControllerObject != null)
		{
			gameController = gameControllerObject.GetComponent <Done_GameController>();
		}
		if (gameController == null)
		{
			Debug.Log ("Cannot find 'GameController' script");
		}
	}

    // destroy the player in contact.
	void OnTriggerEnter (Collider other)
	{
		if (other.tag == "Boundary" || other.tag == "Enemy")
		{
			return;
		}

        // explode this object.
        /*
		if (explosion != null)
		{
			Instantiate(explosion, transform.position, transform.rotation);
		}
        */

        // in contact with player,
		if (other.tag == "Player")
		{
            damagePlayer(other.transform);
            // spawn explostion at player's position.
			Instantiate(playerExplosion, other.transform.position, other.transform.rotation);
			gameController.GameOver();
		}
		
        // add to the score.
		gameController.AddScore(scoreValue);
        // destroy any object in contact and itself.
		//Destroy (other.gameObject);
		//Destroy (gameObject);
	}

    void damagePlayer(Transform player) {
        Transform progressCanvas = transform.FindChild("ProgressCanvas");
        Transform progressBar = progressCanvas.FindChild("ProgressBG");
        Image progressBarImg = progressBar.FindChild("Progress").GetComponent<Image>();
        // if this.mass <= player.mass, do nothing (just bump into each other)
        // if otherwise, dmg the player based on mass difference.
        progressBarImg.fillAmount -= 30f;
        // if player health <= 0, destroy the player.
        if (progressBarImg.fillAmount <= 0) {
            Destroy(player.gameObject);
        }
    }
}