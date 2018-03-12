using UnityEngine;
using System.Collections;

public class FacesPlayer : MonoBehaviour {

	public float rotSpeed = 90f;
    public float attackCycle = 3f;

	Transform player;

    void Start() {
        
    }
	// Update is called once per frame
	void Update () {
		if(player == null) {
			// Find the player's ship!
			GameObject go = GameObject.FindWithTag ("Player");

			if(go != null) {
				player = go.transform;
			}
		}

		// At this point, we've either found the player,
		// or he/she doesn't exist right now.

		if(player == null)
			return;	// Try again next frame!

		// HERE -- we know for sure we have a player. Turn to face it!

		Vector3 dir = player.position - transform.position;
        //Debug.Log(Mathf.Abs(dir.sqrMagnitude));

        // if player is within range, then face the player
        if (Mathf.Abs(dir.sqrMagnitude) <= 10f) {
            // if player within close range try to attack every time cycle.

            dir.Normalize();
            float zAngle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg - 90;
            Quaternion desiredRot = Quaternion.Euler(0, 0, zAngle);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, desiredRot, rotSpeed * Time.deltaTime);
        }

	}
}
