package com.example.eduardoaleixo.phase3app;

import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.IOException;

import flightSystem.FileFormatException;
import flightSystem.FlightSystem;

/**
 * An activity that allows an admin to upload flights from file.
 */
public class LoadFlights extends AppCompatActivity {

    /**
     * Initializes the LoadFlights layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_upload_flights);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
    }

    /**
     * Loads flights when the relevant button is pressed.
     * @param view The view of this activity.
     */
    public void loadFlights(View view){
        EditText filename = (EditText) findViewById(R.id.editText_loadFlights);
        TextView status = (TextView) findViewById(R.id.textView_status);

        try {
            //Get the filename from the screen and try to open the file.
            FlightSystem.getInstance().loadFlights(
                    this.getFilesDir().toString() +
                            "/" + filename.getText().toString());

            status.setText(R.string.success_flights_uploaded);
        } catch (FileFormatException e) {
            status.setText(e.getMessage());
        } catch (FileNotFoundException e){
            status.setText(R.string.file_not_found);
        } catch (Exception e){
            status.setText(e.getMessage());
        }

        //Save the upload
        try {
            FlightSystem.getInstance().save(
                    this.getApplicationContext().getFilesDir().toString());
        } catch (FileNotFoundException f){
            status.setText(R.string.file_not_found);
        } catch (IOException e) {
            status.setText(R.string.upload_not_saved);
        }


    }
}
